ض
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
 �"serve*2.3.02v2.3.0-0-gb36436b0878��
�
conv2d_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_94/kernel
}
$conv2d_94/kernel/Read/ReadVariableOpReadVariableOpconv2d_94/kernel*&
_output_shapes
:*
dtype0
t
conv2d_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_94/bias
m
"conv2d_94/bias/Read/ReadVariableOpReadVariableOpconv2d_94/bias*
_output_shapes
:*
dtype0
�
conv2d_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_95/kernel
}
$conv2d_95/kernel/Read/ReadVariableOpReadVariableOpconv2d_95/kernel*&
_output_shapes
:*
dtype0
t
conv2d_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_95/bias
m
"conv2d_95/bias/Read/ReadVariableOpReadVariableOpconv2d_95/bias*
_output_shapes
:*
dtype0
�
conv2d_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_96/kernel
}
$conv2d_96/kernel/Read/ReadVariableOpReadVariableOpconv2d_96/kernel*&
_output_shapes
:*
dtype0
t
conv2d_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_96/bias
m
"conv2d_96/bias/Read/ReadVariableOpReadVariableOpconv2d_96/bias*
_output_shapes
:*
dtype0
�
conv2d_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_97/kernel
}
$conv2d_97/kernel/Read/ReadVariableOpReadVariableOpconv2d_97/kernel*&
_output_shapes
:*
dtype0
t
conv2d_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_97/bias
m
"conv2d_97/bias/Read/ReadVariableOpReadVariableOpconv2d_97/bias*
_output_shapes
:*
dtype0
�
conv2d_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_98/kernel
}
$conv2d_98/kernel/Read/ReadVariableOpReadVariableOpconv2d_98/kernel*&
_output_shapes
:*
dtype0
t
conv2d_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_98/bias
m
"conv2d_98/bias/Read/ReadVariableOpReadVariableOpconv2d_98/bias*
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
RMSprop/conv2d_94/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_94/kernel/rms
�
0RMSprop/conv2d_94/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_94/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_94/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_94/bias/rms
�
.RMSprop/conv2d_94/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_94/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/conv2d_95/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_95/kernel/rms
�
0RMSprop/conv2d_95/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_95/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_95/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_95/bias/rms
�
.RMSprop/conv2d_95/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_95/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/conv2d_96/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_96/kernel/rms
�
0RMSprop/conv2d_96/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_96/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_96/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_96/bias/rms
�
.RMSprop/conv2d_96/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_96/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/conv2d_97/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_97/kernel/rms
�
0RMSprop/conv2d_97/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_97/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_97/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_97/bias/rms
�
.RMSprop/conv2d_97/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_97/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/conv2d_98/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_98/kernel/rms
�
0RMSprop/conv2d_98/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_98/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_98/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_98/bias/rms
�
.RMSprop/conv2d_98/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_98/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
�B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�B
value�BB�B B�B
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
R
2regularization_losses
3	variables
4trainable_variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
R
<regularization_losses
=	variables
>trainable_variables
?	keras_api
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
�
Xiter
	Ydecay
Zlearning_rate
[momentum
\rho
rms�
rms�
(rms�
)rms�
6rms�
7rms�
Drms�
Erms�
Rrms�
Srms�
 
F
0
1
(2
)3
64
75
D6
E7
R8
S9
F
0
1
(2
)3
64
75
D6
E7
R8
S9
�
regularization_losses
]layer_regularization_losses
^non_trainable_variables
	variables
_metrics

`layers
trainable_variables
alayer_metrics
 
 
 
 
�
regularization_losses
blayer_regularization_losses
cnon_trainable_variables
	variables
dmetrics

elayers
trainable_variables
flayer_metrics
\Z
VARIABLE_VALUEconv2d_94/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_94/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
glayer_regularization_losses
hnon_trainable_variables
	variables
imetrics

jlayers
trainable_variables
klayer_metrics
 
 
 
�
 regularization_losses
llayer_regularization_losses
mnon_trainable_variables
!	variables
nmetrics

olayers
"trainable_variables
player_metrics
 
 
 
�
$regularization_losses
qlayer_regularization_losses
rnon_trainable_variables
%	variables
smetrics

tlayers
&trainable_variables
ulayer_metrics
\Z
VARIABLE_VALUEconv2d_95/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_95/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
*regularization_losses
vlayer_regularization_losses
wnon_trainable_variables
+	variables
xmetrics

ylayers
,trainable_variables
zlayer_metrics
 
 
 
�
.regularization_losses
{layer_regularization_losses
|non_trainable_variables
/	variables
}metrics

~layers
0trainable_variables
layer_metrics
 
 
 
�
2regularization_losses
 �layer_regularization_losses
�non_trainable_variables
3	variables
�metrics
�layers
4trainable_variables
�layer_metrics
\Z
VARIABLE_VALUEconv2d_96/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_96/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
�
8regularization_losses
 �layer_regularization_losses
�non_trainable_variables
9	variables
�metrics
�layers
:trainable_variables
�layer_metrics
 
 
 
�
<regularization_losses
 �layer_regularization_losses
�non_trainable_variables
=	variables
�metrics
�layers
>trainable_variables
�layer_metrics
 
 
 
�
@regularization_losses
 �layer_regularization_losses
�non_trainable_variables
A	variables
�metrics
�layers
Btrainable_variables
�layer_metrics
\Z
VARIABLE_VALUEconv2d_97/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_97/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1
�
Fregularization_losses
 �layer_regularization_losses
�non_trainable_variables
G	variables
�metrics
�layers
Htrainable_variables
�layer_metrics
 
 
 
�
Jregularization_losses
 �layer_regularization_losses
�non_trainable_variables
K	variables
�metrics
�layers
Ltrainable_variables
�layer_metrics
 
 
 
�
Nregularization_losses
 �layer_regularization_losses
�non_trainable_variables
O	variables
�metrics
�layers
Ptrainable_variables
�layer_metrics
\Z
VARIABLE_VALUEconv2d_98/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_98/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
�
Tregularization_losses
 �layer_regularization_losses
�non_trainable_variables
U	variables
�metrics
�layers
Vtrainable_variables
�layer_metrics
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

�0
�1
n
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
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUERMSprop/conv2d_94/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_94/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_95/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_95/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_96/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_96/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_97/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_97/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_98/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_98/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_27Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_27conv2d_94/kernelconv2d_94/biasconv2d_95/kernelconv2d_95/biasconv2d_96/kernelconv2d_96/biasconv2d_97/kernelconv2d_97/biasconv2d_98/kernelconv2d_98/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_144555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_94/kernel/Read/ReadVariableOp"conv2d_94/bias/Read/ReadVariableOp$conv2d_95/kernel/Read/ReadVariableOp"conv2d_95/bias/Read/ReadVariableOp$conv2d_96/kernel/Read/ReadVariableOp"conv2d_96/bias/Read/ReadVariableOp$conv2d_97/kernel/Read/ReadVariableOp"conv2d_97/bias/Read/ReadVariableOp$conv2d_98/kernel/Read/ReadVariableOp"conv2d_98/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0RMSprop/conv2d_94/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_94/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_95/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_95/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_96/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_96/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_97/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_97/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_98/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_98/bias/rms/Read/ReadVariableOpConst**
Tin#
!2	*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_144949
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_94/kernelconv2d_94/biasconv2d_95/kernelconv2d_95/biasconv2d_96/kernelconv2d_96/biasconv2d_97/kernelconv2d_97/biasconv2d_98/kernelconv2d_98/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/conv2d_94/kernel/rmsRMSprop/conv2d_94/bias/rmsRMSprop/conv2d_95/kernel/rmsRMSprop/conv2d_95/bias/rmsRMSprop/conv2d_96/kernel/rmsRMSprop/conv2d_96/bias/rmsRMSprop/conv2d_97/kernel/rmsRMSprop/conv2d_97/bias/rmsRMSprop/conv2d_98/kernel/rmsRMSprop/conv2d_98/bias/rms*)
Tin"
 2*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_145046��
�
i
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_144086

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
�
N
2__inference_zero_padding2d_84_layer_call_fn_144092

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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_1440862
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
N
2__inference_zero_padding2d_86_layer_call_fn_144142

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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_1441362
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
N
2__inference_zero_padding2d_87_layer_call_fn_144174

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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_1441682
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
E__inference_conv2d_97_layer_call_and_return_conditional_losses_144810

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_144555
input_27
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_27unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_1440792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_27
�	
�
E__inference_conv2d_98_layer_call_and_return_conditional_losses_144338

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_144187

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
valueB"      2
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
.__inference_functional_53_layer_call_fn_144520
input_27
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_27unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_53_layer_call_and_return_conditional_losses_1444972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_27
�
i
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_144111

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
�

*__inference_conv2d_95_layer_call_fn_144779

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
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1442512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  2

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

*__inference_conv2d_96_layer_call_fn_144799

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
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1442802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������""::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�

*__inference_conv2d_94_layer_call_fn_144759

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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1442222
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
�	
�
E__inference_conv2d_94_layer_call_and_return_conditional_losses_144750

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
M
1__inference_max_pooling2d_35_layer_call_fn_144129

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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_1441232
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
E__inference_conv2d_96_layer_call_and_return_conditional_losses_144790

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������"":::W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�e
�
!__inference__wrapped_model_144079
input_27:
6functional_53_conv2d_94_conv2d_readvariableop_resource;
7functional_53_conv2d_94_biasadd_readvariableop_resource:
6functional_53_conv2d_95_conv2d_readvariableop_resource;
7functional_53_conv2d_95_biasadd_readvariableop_resource:
6functional_53_conv2d_96_conv2d_readvariableop_resource;
7functional_53_conv2d_96_biasadd_readvariableop_resource:
6functional_53_conv2d_97_conv2d_readvariableop_resource;
7functional_53_conv2d_97_biasadd_readvariableop_resource:
6functional_53_conv2d_98_conv2d_readvariableop_resource;
7functional_53_conv2d_98_biasadd_readvariableop_resource
identity��
,functional_53/zero_padding2d_84/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_53/zero_padding2d_84/Pad/paddings�
#functional_53/zero_padding2d_84/PadPadinput_275functional_53/zero_padding2d_84/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2%
#functional_53/zero_padding2d_84/Pad�
-functional_53/conv2d_94/Conv2D/ReadVariableOpReadVariableOp6functional_53_conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_53/conv2d_94/Conv2D/ReadVariableOp�
functional_53/conv2d_94/Conv2DConv2D,functional_53/zero_padding2d_84/Pad:output:05functional_53/conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2 
functional_53/conv2d_94/Conv2D�
.functional_53/conv2d_94/BiasAdd/ReadVariableOpReadVariableOp7functional_53_conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_53/conv2d_94/BiasAdd/ReadVariableOp�
functional_53/conv2d_94/BiasAddBiasAdd'functional_53/conv2d_94/Conv2D:output:06functional_53/conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
functional_53/conv2d_94/BiasAdd�
functional_53/conv2d_94/ReluRelu(functional_53/conv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_53/conv2d_94/Relu�
&functional_53/max_pooling2d_34/MaxPoolMaxPool*functional_53/conv2d_94/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
2(
&functional_53/max_pooling2d_34/MaxPool�
,functional_53/zero_padding2d_85/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_53/zero_padding2d_85/Pad/paddings�
#functional_53/zero_padding2d_85/PadPad/functional_53/max_pooling2d_34/MaxPool:output:05functional_53/zero_padding2d_85/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2%
#functional_53/zero_padding2d_85/Pad�
-functional_53/conv2d_95/Conv2D/ReadVariableOpReadVariableOp6functional_53_conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_53/conv2d_95/Conv2D/ReadVariableOp�
functional_53/conv2d_95/Conv2DConv2D,functional_53/zero_padding2d_85/Pad:output:05functional_53/conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2 
functional_53/conv2d_95/Conv2D�
.functional_53/conv2d_95/BiasAdd/ReadVariableOpReadVariableOp7functional_53_conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_53/conv2d_95/BiasAdd/ReadVariableOp�
functional_53/conv2d_95/BiasAddBiasAdd'functional_53/conv2d_95/Conv2D:output:06functional_53/conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
functional_53/conv2d_95/BiasAdd�
functional_53/conv2d_95/ReluRelu(functional_53/conv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_53/conv2d_95/Relu�
&functional_53/max_pooling2d_35/MaxPoolMaxPool*functional_53/conv2d_95/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
2(
&functional_53/max_pooling2d_35/MaxPool�
,functional_53/zero_padding2d_86/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_53/zero_padding2d_86/Pad/paddings�
#functional_53/zero_padding2d_86/PadPad/functional_53/max_pooling2d_35/MaxPool:output:05functional_53/zero_padding2d_86/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2%
#functional_53/zero_padding2d_86/Pad�
-functional_53/conv2d_96/Conv2D/ReadVariableOpReadVariableOp6functional_53_conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_53/conv2d_96/Conv2D/ReadVariableOp�
functional_53/conv2d_96/Conv2DConv2D,functional_53/zero_padding2d_86/Pad:output:05functional_53/conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2 
functional_53/conv2d_96/Conv2D�
.functional_53/conv2d_96/BiasAdd/ReadVariableOpReadVariableOp7functional_53_conv2d_96_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_53/conv2d_96/BiasAdd/ReadVariableOp�
functional_53/conv2d_96/BiasAddBiasAdd'functional_53/conv2d_96/Conv2D:output:06functional_53/conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
functional_53/conv2d_96/BiasAdd�
functional_53/conv2d_96/ReluRelu(functional_53/conv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_53/conv2d_96/Relu�
$functional_53/up_sampling2d_34/ShapeShape*functional_53/conv2d_96/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_53/up_sampling2d_34/Shape�
2functional_53/up_sampling2d_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2functional_53/up_sampling2d_34/strided_slice/stack�
4functional_53/up_sampling2d_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_53/up_sampling2d_34/strided_slice/stack_1�
4functional_53/up_sampling2d_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_53/up_sampling2d_34/strided_slice/stack_2�
,functional_53/up_sampling2d_34/strided_sliceStridedSlice-functional_53/up_sampling2d_34/Shape:output:0;functional_53/up_sampling2d_34/strided_slice/stack:output:0=functional_53/up_sampling2d_34/strided_slice/stack_1:output:0=functional_53/up_sampling2d_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2.
,functional_53/up_sampling2d_34/strided_slice�
$functional_53/up_sampling2d_34/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2&
$functional_53/up_sampling2d_34/Const�
"functional_53/up_sampling2d_34/mulMul5functional_53/up_sampling2d_34/strided_slice:output:0-functional_53/up_sampling2d_34/Const:output:0*
T0*
_output_shapes
:2$
"functional_53/up_sampling2d_34/mul�
;functional_53/up_sampling2d_34/resize/ResizeNearestNeighborResizeNearestNeighbor*functional_53/conv2d_96/Relu:activations:0&functional_53/up_sampling2d_34/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2=
;functional_53/up_sampling2d_34/resize/ResizeNearestNeighbor�
,functional_53/zero_padding2d_87/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_53/zero_padding2d_87/Pad/paddings�
#functional_53/zero_padding2d_87/PadPadLfunctional_53/up_sampling2d_34/resize/ResizeNearestNeighbor:resized_images:05functional_53/zero_padding2d_87/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2%
#functional_53/zero_padding2d_87/Pad�
-functional_53/conv2d_97/Conv2D/ReadVariableOpReadVariableOp6functional_53_conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_53/conv2d_97/Conv2D/ReadVariableOp�
functional_53/conv2d_97/Conv2DConv2D,functional_53/zero_padding2d_87/Pad:output:05functional_53/conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2 
functional_53/conv2d_97/Conv2D�
.functional_53/conv2d_97/BiasAdd/ReadVariableOpReadVariableOp7functional_53_conv2d_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_53/conv2d_97/BiasAdd/ReadVariableOp�
functional_53/conv2d_97/BiasAddBiasAdd'functional_53/conv2d_97/Conv2D:output:06functional_53/conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
functional_53/conv2d_97/BiasAdd�
functional_53/conv2d_97/ReluRelu(functional_53/conv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_53/conv2d_97/Relu�
$functional_53/up_sampling2d_35/ShapeShape*functional_53/conv2d_97/Relu:activations:0*
T0*
_output_shapes
:2&
$functional_53/up_sampling2d_35/Shape�
2functional_53/up_sampling2d_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2functional_53/up_sampling2d_35/strided_slice/stack�
4functional_53/up_sampling2d_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_53/up_sampling2d_35/strided_slice/stack_1�
4functional_53/up_sampling2d_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_53/up_sampling2d_35/strided_slice/stack_2�
,functional_53/up_sampling2d_35/strided_sliceStridedSlice-functional_53/up_sampling2d_35/Shape:output:0;functional_53/up_sampling2d_35/strided_slice/stack:output:0=functional_53/up_sampling2d_35/strided_slice/stack_1:output:0=functional_53/up_sampling2d_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2.
,functional_53/up_sampling2d_35/strided_slice�
$functional_53/up_sampling2d_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2&
$functional_53/up_sampling2d_35/Const�
"functional_53/up_sampling2d_35/mulMul5functional_53/up_sampling2d_35/strided_slice:output:0-functional_53/up_sampling2d_35/Const:output:0*
T0*
_output_shapes
:2$
"functional_53/up_sampling2d_35/mul�
;functional_53/up_sampling2d_35/resize/ResizeNearestNeighborResizeNearestNeighbor*functional_53/conv2d_97/Relu:activations:0&functional_53/up_sampling2d_35/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2=
;functional_53/up_sampling2d_35/resize/ResizeNearestNeighbor�
,functional_53/zero_padding2d_88/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_53/zero_padding2d_88/Pad/paddings�
#functional_53/zero_padding2d_88/PadPadLfunctional_53/up_sampling2d_35/resize/ResizeNearestNeighbor:resized_images:05functional_53/zero_padding2d_88/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2%
#functional_53/zero_padding2d_88/Pad�
-functional_53/conv2d_98/Conv2D/ReadVariableOpReadVariableOp6functional_53_conv2d_98_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_53/conv2d_98/Conv2D/ReadVariableOp�
functional_53/conv2d_98/Conv2DConv2D,functional_53/zero_padding2d_88/Pad:output:05functional_53/conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2 
functional_53/conv2d_98/Conv2D�
.functional_53/conv2d_98/BiasAdd/ReadVariableOpReadVariableOp7functional_53_conv2d_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_53/conv2d_98/BiasAdd/ReadVariableOp�
functional_53/conv2d_98/BiasAddBiasAdd'functional_53/conv2d_98/Conv2D:output:06functional_53/conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
functional_53/conv2d_98/BiasAdd�
functional_53/conv2d_98/ReluRelu(functional_53/conv2d_98/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_53/conv2d_98/Relu�
IdentityIdentity*functional_53/conv2d_98/Relu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  :::::::::::Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_27
�
i
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_144136

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
�{
�
"__inference__traced_restore_145046
file_prefix%
!assignvariableop_conv2d_94_kernel%
!assignvariableop_1_conv2d_94_bias'
#assignvariableop_2_conv2d_95_kernel%
!assignvariableop_3_conv2d_95_bias'
#assignvariableop_4_conv2d_96_kernel%
!assignvariableop_5_conv2d_96_bias'
#assignvariableop_6_conv2d_97_kernel%
!assignvariableop_7_conv2d_97_bias'
#assignvariableop_8_conv2d_98_kernel%
!assignvariableop_9_conv2d_98_bias$
 assignvariableop_10_rmsprop_iter%
!assignvariableop_11_rmsprop_decay-
)assignvariableop_12_rmsprop_learning_rate(
$assignvariableop_13_rmsprop_momentum#
assignvariableop_14_rmsprop_rho
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_14
0assignvariableop_19_rmsprop_conv2d_94_kernel_rms2
.assignvariableop_20_rmsprop_conv2d_94_bias_rms4
0assignvariableop_21_rmsprop_conv2d_95_kernel_rms2
.assignvariableop_22_rmsprop_conv2d_95_bias_rms4
0assignvariableop_23_rmsprop_conv2d_96_kernel_rms2
.assignvariableop_24_rmsprop_conv2d_96_bias_rms4
0assignvariableop_25_rmsprop_conv2d_97_kernel_rms2
.assignvariableop_26_rmsprop_conv2d_97_bias_rms4
0assignvariableop_27_rmsprop_conv2d_98_kernel_rms2
.assignvariableop_28_rmsprop_conv2d_98_bias_rms
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_94_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_94_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_95_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_95_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_96_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_96_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_97_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_97_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_98_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_98_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp assignvariableop_10_rmsprop_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_rmsprop_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_rmsprop_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_rmsprop_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_rmsprop_rhoIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_rmsprop_conv2d_94_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_rmsprop_conv2d_94_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp0assignvariableop_21_rmsprop_conv2d_95_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_rmsprop_conv2d_95_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp0assignvariableop_23_rmsprop_conv2d_96_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp.assignvariableop_24_rmsprop_conv2d_96_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp0assignvariableop_25_rmsprop_conv2d_97_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp.assignvariableop_26_rmsprop_conv2d_97_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_rmsprop_conv2d_98_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_rmsprop_conv2d_98_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29�
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*�
_input_shapesx
v: :::::::::::::::::::::::::::::2$
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
AssignVariableOp_28AssignVariableOp_282(
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
�B
�
__inference__traced_save_144949
file_prefix/
+savev2_conv2d_94_kernel_read_readvariableop-
)savev2_conv2d_94_bias_read_readvariableop/
+savev2_conv2d_95_kernel_read_readvariableop-
)savev2_conv2d_95_bias_read_readvariableop/
+savev2_conv2d_96_kernel_read_readvariableop-
)savev2_conv2d_96_bias_read_readvariableop/
+savev2_conv2d_97_kernel_read_readvariableop-
)savev2_conv2d_97_bias_read_readvariableop/
+savev2_conv2d_98_kernel_read_readvariableop-
)savev2_conv2d_98_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_rmsprop_conv2d_94_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_94_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_95_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_95_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_96_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_96_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_97_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_97_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_98_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_98_bias_rms_read_readvariableop
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
value3B1 B+_temp_fcf58128c42d42089828725e5625d4f1/part2	
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_94_kernel_read_readvariableop)savev2_conv2d_94_bias_read_readvariableop+savev2_conv2d_95_kernel_read_readvariableop)savev2_conv2d_95_bias_read_readvariableop+savev2_conv2d_96_kernel_read_readvariableop)savev2_conv2d_96_bias_read_readvariableop+savev2_conv2d_97_kernel_read_readvariableop)savev2_conv2d_97_bias_read_readvariableop+savev2_conv2d_98_kernel_read_readvariableop)savev2_conv2d_98_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_rmsprop_conv2d_94_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_94_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_95_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_95_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_96_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_96_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_97_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_97_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_98_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_98_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::: : : : : : : : : ::::::::::: 2(
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�:
�
I__inference_functional_53_layer_call_and_return_conditional_losses_144434

inputs
conv2d_94_144400
conv2d_94_144402
conv2d_95_144407
conv2d_95_144409
conv2d_96_144414
conv2d_96_144416
conv2d_97_144421
conv2d_97_144423
conv2d_98_144428
conv2d_98_144430
identity��!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�
!zero_padding2d_84/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_1440862#
!zero_padding2d_84/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_84/PartitionedCall:output:0conv2d_94_144400conv2d_94_144402*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1442222#
!conv2d_94/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_1440982"
 max_pooling2d_34/PartitionedCall�
!zero_padding2d_85/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_1441112#
!zero_padding2d_85/PartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_85/PartitionedCall:output:0conv2d_95_144407conv2d_95_144409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1442512#
!conv2d_95/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_1441232"
 max_pooling2d_35/PartitionedCall�
!zero_padding2d_86/PartitionedCallPartitionedCall)max_pooling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_1441362#
!zero_padding2d_86/PartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_86/PartitionedCall:output:0conv2d_96_144414conv2d_96_144416*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1442802#
!conv2d_96/StatefulPartitionedCall�
 up_sampling2d_34/PartitionedCallPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_1441552"
 up_sampling2d_34/PartitionedCall�
!zero_padding2d_87/PartitionedCallPartitionedCall)up_sampling2d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_1441682#
!zero_padding2d_87/PartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_87/PartitionedCall:output:0conv2d_97_144421conv2d_97_144423*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1443092#
!conv2d_97/StatefulPartitionedCall�
 up_sampling2d_35/PartitionedCallPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_1441872"
 up_sampling2d_35/PartitionedCall�
!zero_padding2d_88/PartitionedCallPartitionedCall)up_sampling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_1442002#
!zero_padding2d_88/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_88/PartitionedCall:output:0conv2d_98_144428conv2d_98_144430*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_1443382#
!conv2d_98/StatefulPartitionedCall�
IdentityIdentity*conv2d_98/StatefulPartitionedCall:output:0"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�R
�
I__inference_functional_53_layer_call_and_return_conditional_losses_144622

inputs,
(conv2d_94_conv2d_readvariableop_resource-
)conv2d_94_biasadd_readvariableop_resource,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource,
(conv2d_97_conv2d_readvariableop_resource-
)conv2d_97_biasadd_readvariableop_resource,
(conv2d_98_conv2d_readvariableop_resource-
)conv2d_98_biasadd_readvariableop_resource
identity��
zero_padding2d_84/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_84/Pad/paddings�
zero_padding2d_84/PadPadinputs'zero_padding2d_84/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_84/Pad�
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_94/Conv2D/ReadVariableOp�
conv2d_94/Conv2DConv2Dzero_padding2d_84/Pad:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_94/Conv2D�
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp�
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_94/Relu�
max_pooling2d_34/MaxPoolMaxPoolconv2d_94/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_34/MaxPool�
zero_padding2d_85/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_85/Pad/paddings�
zero_padding2d_85/PadPad!max_pooling2d_34/MaxPool:output:0'zero_padding2d_85/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_85/Pad�
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_95/Conv2D/ReadVariableOp�
conv2d_95/Conv2DConv2Dzero_padding2d_85/Pad:output:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_95/Conv2D�
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp�
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_95/Relu�
max_pooling2d_35/MaxPoolMaxPoolconv2d_95/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_35/MaxPool�
zero_padding2d_86/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_86/Pad/paddings�
zero_padding2d_86/PadPad!max_pooling2d_35/MaxPool:output:0'zero_padding2d_86/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_86/Pad�
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_96/Conv2D/ReadVariableOp�
conv2d_96/Conv2DConv2Dzero_padding2d_86/Pad:output:0'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_96/Conv2D�
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp�
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_96/BiasAdd~
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_96/Relu|
up_sampling2d_34/ShapeShapeconv2d_96/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_34/Shape�
$up_sampling2d_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_34/strided_slice/stack�
&up_sampling2d_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_34/strided_slice/stack_1�
&up_sampling2d_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_34/strided_slice/stack_2�
up_sampling2d_34/strided_sliceStridedSliceup_sampling2d_34/Shape:output:0-up_sampling2d_34/strided_slice/stack:output:0/up_sampling2d_34/strided_slice/stack_1:output:0/up_sampling2d_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_34/strided_slice�
up_sampling2d_34/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_34/Const�
up_sampling2d_34/mulMul'up_sampling2d_34/strided_slice:output:0up_sampling2d_34/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_34/mul�
-up_sampling2d_34/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_96/Relu:activations:0up_sampling2d_34/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2/
-up_sampling2d_34/resize/ResizeNearestNeighbor�
zero_padding2d_87/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_87/Pad/paddings�
zero_padding2d_87/PadPad>up_sampling2d_34/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_87/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_87/Pad�
conv2d_97/Conv2D/ReadVariableOpReadVariableOp(conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_97/Conv2D/ReadVariableOp�
conv2d_97/Conv2DConv2Dzero_padding2d_87/Pad:output:0'conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_97/Conv2D�
 conv2d_97/BiasAdd/ReadVariableOpReadVariableOp)conv2d_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_97/BiasAdd/ReadVariableOp�
conv2d_97/BiasAddBiasAddconv2d_97/Conv2D:output:0(conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_97/BiasAdd~
conv2d_97/ReluReluconv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_97/Relu|
up_sampling2d_35/ShapeShapeconv2d_97/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_35/Shape�
$up_sampling2d_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_35/strided_slice/stack�
&up_sampling2d_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_35/strided_slice/stack_1�
&up_sampling2d_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_35/strided_slice/stack_2�
up_sampling2d_35/strided_sliceStridedSliceup_sampling2d_35/Shape:output:0-up_sampling2d_35/strided_slice/stack:output:0/up_sampling2d_35/strided_slice/stack_1:output:0/up_sampling2d_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_35/strided_slice�
up_sampling2d_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_35/Const�
up_sampling2d_35/mulMul'up_sampling2d_35/strided_slice:output:0up_sampling2d_35/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_35/mul�
-up_sampling2d_35/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_97/Relu:activations:0up_sampling2d_35/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2/
-up_sampling2d_35/resize/ResizeNearestNeighbor�
zero_padding2d_88/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_88/Pad/paddings�
zero_padding2d_88/PadPad>up_sampling2d_35/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_88/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_88/Pad�
conv2d_98/Conv2D/ReadVariableOpReadVariableOp(conv2d_98_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_98/Conv2D/ReadVariableOp�
conv2d_98/Conv2DConv2Dzero_padding2d_88/Pad:output:0'conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_98/Conv2D�
 conv2d_98/BiasAdd/ReadVariableOpReadVariableOp)conv2d_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_98/BiasAdd/ReadVariableOp�
conv2d_98/BiasAddBiasAddconv2d_98/Conv2D:output:0(conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_98/BiasAdd~
conv2d_98/ReluReluconv2d_98/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_98/Relux
IdentityIdentityconv2d_98/Relu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  :::::::::::W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�	
�
.__inference_functional_53_layer_call_fn_144739

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_53_layer_call_and_return_conditional_losses_1444972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_144123

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
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
�
N
2__inference_zero_padding2d_88_layer_call_fn_144206

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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_1442002
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
h
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_144098

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
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
i
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_144200

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
�
E__inference_conv2d_95_layer_call_and_return_conditional_losses_144251

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������"":::W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_144155

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
valueB"      2
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
�:
�
I__inference_functional_53_layer_call_and_return_conditional_losses_144355
input_27
conv2d_94_144233
conv2d_94_144235
conv2d_95_144262
conv2d_95_144264
conv2d_96_144291
conv2d_96_144293
conv2d_97_144320
conv2d_97_144322
conv2d_98_144349
conv2d_98_144351
identity��!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�
!zero_padding2d_84/PartitionedCallPartitionedCallinput_27*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_1440862#
!zero_padding2d_84/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_84/PartitionedCall:output:0conv2d_94_144233conv2d_94_144235*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1442222#
!conv2d_94/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_1440982"
 max_pooling2d_34/PartitionedCall�
!zero_padding2d_85/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_1441112#
!zero_padding2d_85/PartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_85/PartitionedCall:output:0conv2d_95_144262conv2d_95_144264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1442512#
!conv2d_95/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_1441232"
 max_pooling2d_35/PartitionedCall�
!zero_padding2d_86/PartitionedCallPartitionedCall)max_pooling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_1441362#
!zero_padding2d_86/PartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_86/PartitionedCall:output:0conv2d_96_144291conv2d_96_144293*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1442802#
!conv2d_96/StatefulPartitionedCall�
 up_sampling2d_34/PartitionedCallPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_1441552"
 up_sampling2d_34/PartitionedCall�
!zero_padding2d_87/PartitionedCallPartitionedCall)up_sampling2d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_1441682#
!zero_padding2d_87/PartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_87/PartitionedCall:output:0conv2d_97_144320conv2d_97_144322*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1443092#
!conv2d_97/StatefulPartitionedCall�
 up_sampling2d_35/PartitionedCallPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_1441872"
 up_sampling2d_35/PartitionedCall�
!zero_padding2d_88/PartitionedCallPartitionedCall)up_sampling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_1442002#
!zero_padding2d_88/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_88/PartitionedCall:output:0conv2d_98_144349conv2d_98_144351*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_1443382#
!conv2d_98/StatefulPartitionedCall�
IdentityIdentity*conv2d_98/StatefulPartitionedCall:output:0"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_27
�
M
1__inference_up_sampling2d_34_layer_call_fn_144161

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
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_1441552
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
.__inference_functional_53_layer_call_fn_144714

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_53_layer_call_and_return_conditional_losses_1444342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

*__inference_conv2d_98_layer_call_fn_144839

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
GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_1443382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
E__inference_conv2d_97_layer_call_and_return_conditional_losses_144309

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
E__inference_conv2d_96_layer_call_and_return_conditional_losses_144280

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������"":::W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_34_layer_call_fn_144104

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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_1440982
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

*__inference_conv2d_97_layer_call_fn_144819

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
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1443092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�:
�
I__inference_functional_53_layer_call_and_return_conditional_losses_144393
input_27
conv2d_94_144359
conv2d_94_144361
conv2d_95_144366
conv2d_95_144368
conv2d_96_144373
conv2d_96_144375
conv2d_97_144380
conv2d_97_144382
conv2d_98_144387
conv2d_98_144389
identity��!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�
!zero_padding2d_84/PartitionedCallPartitionedCallinput_27*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_1440862#
!zero_padding2d_84/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_84/PartitionedCall:output:0conv2d_94_144359conv2d_94_144361*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1442222#
!conv2d_94/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_1440982"
 max_pooling2d_34/PartitionedCall�
!zero_padding2d_85/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_1441112#
!zero_padding2d_85/PartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_85/PartitionedCall:output:0conv2d_95_144366conv2d_95_144368*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1442512#
!conv2d_95/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_1441232"
 max_pooling2d_35/PartitionedCall�
!zero_padding2d_86/PartitionedCallPartitionedCall)max_pooling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_1441362#
!zero_padding2d_86/PartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_86/PartitionedCall:output:0conv2d_96_144373conv2d_96_144375*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1442802#
!conv2d_96/StatefulPartitionedCall�
 up_sampling2d_34/PartitionedCallPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_1441552"
 up_sampling2d_34/PartitionedCall�
!zero_padding2d_87/PartitionedCallPartitionedCall)up_sampling2d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_1441682#
!zero_padding2d_87/PartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_87/PartitionedCall:output:0conv2d_97_144380conv2d_97_144382*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1443092#
!conv2d_97/StatefulPartitionedCall�
 up_sampling2d_35/PartitionedCallPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_1441872"
 up_sampling2d_35/PartitionedCall�
!zero_padding2d_88/PartitionedCallPartitionedCall)up_sampling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_1442002#
!zero_padding2d_88/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_88/PartitionedCall:output:0conv2d_98_144387conv2d_98_144389*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_1443382#
!conv2d_98/StatefulPartitionedCall�
IdentityIdentity*conv2d_98/StatefulPartitionedCall:output:0"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_27
�	
�
E__inference_conv2d_98_layer_call_and_return_conditional_losses_144830

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
E__inference_conv2d_94_layer_call_and_return_conditional_losses_144222

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
N
2__inference_zero_padding2d_85_layer_call_fn_144117

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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_1441112
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
M
1__inference_up_sampling2d_35_layer_call_fn_144193

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
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_1441872
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
.__inference_functional_53_layer_call_fn_144457
input_27
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_27unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_functional_53_layer_call_and_return_conditional_losses_1444342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_27
�
i
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_144168

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
�
E__inference_conv2d_95_layer_call_and_return_conditional_losses_144770

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������"":::W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�R
�
I__inference_functional_53_layer_call_and_return_conditional_losses_144689

inputs,
(conv2d_94_conv2d_readvariableop_resource-
)conv2d_94_biasadd_readvariableop_resource,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource,
(conv2d_97_conv2d_readvariableop_resource-
)conv2d_97_biasadd_readvariableop_resource,
(conv2d_98_conv2d_readvariableop_resource-
)conv2d_98_biasadd_readvariableop_resource
identity��
zero_padding2d_84/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_84/Pad/paddings�
zero_padding2d_84/PadPadinputs'zero_padding2d_84/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_84/Pad�
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_94/Conv2D/ReadVariableOp�
conv2d_94/Conv2DConv2Dzero_padding2d_84/Pad:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_94/Conv2D�
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp�
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_94/Relu�
max_pooling2d_34/MaxPoolMaxPoolconv2d_94/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_34/MaxPool�
zero_padding2d_85/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_85/Pad/paddings�
zero_padding2d_85/PadPad!max_pooling2d_34/MaxPool:output:0'zero_padding2d_85/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_85/Pad�
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_95/Conv2D/ReadVariableOp�
conv2d_95/Conv2DConv2Dzero_padding2d_85/Pad:output:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_95/Conv2D�
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp�
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_95/Relu�
max_pooling2d_35/MaxPoolMaxPoolconv2d_95/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingSAME*
strides
2
max_pooling2d_35/MaxPool�
zero_padding2d_86/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_86/Pad/paddings�
zero_padding2d_86/PadPad!max_pooling2d_35/MaxPool:output:0'zero_padding2d_86/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_86/Pad�
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_96/Conv2D/ReadVariableOp�
conv2d_96/Conv2DConv2Dzero_padding2d_86/Pad:output:0'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_96/Conv2D�
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp�
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_96/BiasAdd~
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_96/Relu|
up_sampling2d_34/ShapeShapeconv2d_96/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_34/Shape�
$up_sampling2d_34/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_34/strided_slice/stack�
&up_sampling2d_34/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_34/strided_slice/stack_1�
&up_sampling2d_34/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_34/strided_slice/stack_2�
up_sampling2d_34/strided_sliceStridedSliceup_sampling2d_34/Shape:output:0-up_sampling2d_34/strided_slice/stack:output:0/up_sampling2d_34/strided_slice/stack_1:output:0/up_sampling2d_34/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_34/strided_slice�
up_sampling2d_34/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_34/Const�
up_sampling2d_34/mulMul'up_sampling2d_34/strided_slice:output:0up_sampling2d_34/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_34/mul�
-up_sampling2d_34/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_96/Relu:activations:0up_sampling2d_34/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2/
-up_sampling2d_34/resize/ResizeNearestNeighbor�
zero_padding2d_87/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_87/Pad/paddings�
zero_padding2d_87/PadPad>up_sampling2d_34/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_87/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_87/Pad�
conv2d_97/Conv2D/ReadVariableOpReadVariableOp(conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_97/Conv2D/ReadVariableOp�
conv2d_97/Conv2DConv2Dzero_padding2d_87/Pad:output:0'conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_97/Conv2D�
 conv2d_97/BiasAdd/ReadVariableOpReadVariableOp)conv2d_97_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_97/BiasAdd/ReadVariableOp�
conv2d_97/BiasAddBiasAddconv2d_97/Conv2D:output:0(conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_97/BiasAdd~
conv2d_97/ReluReluconv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_97/Relu|
up_sampling2d_35/ShapeShapeconv2d_97/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_35/Shape�
$up_sampling2d_35/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_35/strided_slice/stack�
&up_sampling2d_35/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_35/strided_slice/stack_1�
&up_sampling2d_35/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_35/strided_slice/stack_2�
up_sampling2d_35/strided_sliceStridedSliceup_sampling2d_35/Shape:output:0-up_sampling2d_35/strided_slice/stack:output:0/up_sampling2d_35/strided_slice/stack_1:output:0/up_sampling2d_35/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_35/strided_slice�
up_sampling2d_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_35/Const�
up_sampling2d_35/mulMul'up_sampling2d_35/strided_slice:output:0up_sampling2d_35/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_35/mul�
-up_sampling2d_35/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_97/Relu:activations:0up_sampling2d_35/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2/
-up_sampling2d_35/resize/ResizeNearestNeighbor�
zero_padding2d_88/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_88/Pad/paddings�
zero_padding2d_88/PadPad>up_sampling2d_35/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_88/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_88/Pad�
conv2d_98/Conv2D/ReadVariableOpReadVariableOp(conv2d_98_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_98/Conv2D/ReadVariableOp�
conv2d_98/Conv2DConv2Dzero_padding2d_88/Pad:output:0'conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_98/Conv2D�
 conv2d_98/BiasAdd/ReadVariableOpReadVariableOp)conv2d_98_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_98/BiasAdd/ReadVariableOp�
conv2d_98/BiasAddBiasAddconv2d_98/Conv2D:output:0(conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_98/BiasAdd~
conv2d_98/ReluReluconv2d_98/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_98/Relux
IdentityIdentityconv2d_98/Relu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  :::::::::::W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�:
�
I__inference_functional_53_layer_call_and_return_conditional_losses_144497

inputs
conv2d_94_144463
conv2d_94_144465
conv2d_95_144470
conv2d_95_144472
conv2d_96_144477
conv2d_96_144479
conv2d_97_144484
conv2d_97_144486
conv2d_98_144491
conv2d_98_144493
identity��!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�
!zero_padding2d_84/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_1440862#
!zero_padding2d_84/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_84/PartitionedCall:output:0conv2d_94_144463conv2d_94_144465*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1442222#
!conv2d_94/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_1440982"
 max_pooling2d_34/PartitionedCall�
!zero_padding2d_85/PartitionedCallPartitionedCall)max_pooling2d_34/PartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_1441112#
!zero_padding2d_85/PartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_85/PartitionedCall:output:0conv2d_95_144470conv2d_95_144472*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1442512#
!conv2d_95/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_1441232"
 max_pooling2d_35/PartitionedCall�
!zero_padding2d_86/PartitionedCallPartitionedCall)max_pooling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_1441362#
!zero_padding2d_86/PartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_86/PartitionedCall:output:0conv2d_96_144477conv2d_96_144479*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1442802#
!conv2d_96/StatefulPartitionedCall�
 up_sampling2d_34/PartitionedCallPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_1441552"
 up_sampling2d_34/PartitionedCall�
!zero_padding2d_87/PartitionedCallPartitionedCall)up_sampling2d_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_1441682#
!zero_padding2d_87/PartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_87/PartitionedCall:output:0conv2d_97_144484conv2d_97_144486*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1443092#
!conv2d_97/StatefulPartitionedCall�
 up_sampling2d_35/PartitionedCallPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_1441872"
 up_sampling2d_35/PartitionedCall�
!zero_padding2d_88/PartitionedCallPartitionedCall)up_sampling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_1442002#
!zero_padding2d_88/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_88/PartitionedCall:output:0conv2d_98_144491conv2d_98_144493*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_1443382#
!conv2d_98/StatefulPartitionedCall�
IdentityIdentity*conv2d_98/StatefulPartitionedCall:output:0"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:���������  ::::::::::2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall:W S
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
input_279
serving_default_input_27:0���������  E
	conv2d_988
StatefulPartitionedCall:0���������  tensorflow/serving/predict:�
�{
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�w
_tf_keras_network�w{"class_name": "Functional", "name": "functional_53", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_53", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_27"}, "name": "input_27", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_84", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_84", "inbound_nodes": [[["input_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_94", "inbound_nodes": [[["zero_padding2d_84", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_34", "inbound_nodes": [[["conv2d_94", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_85", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_85", "inbound_nodes": [[["max_pooling2d_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_95", "inbound_nodes": [[["zero_padding2d_85", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_35", "inbound_nodes": [[["conv2d_95", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_86", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_86", "inbound_nodes": [[["max_pooling2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_96", "inbound_nodes": [[["zero_padding2d_86", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_34", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_34", "inbound_nodes": [[["conv2d_96", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_87", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_87", "inbound_nodes": [[["up_sampling2d_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_97", "inbound_nodes": [[["zero_padding2d_87", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_35", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_35", "inbound_nodes": [[["conv2d_97", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_88", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_88", "inbound_nodes": [[["up_sampling2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_98", "inbound_nodes": [[["zero_padding2d_88", 0, 0, {}]]]}], "input_layers": [["input_27", 0, 0]], "output_layers": [["conv2d_98", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_53", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_27"}, "name": "input_27", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_84", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_84", "inbound_nodes": [[["input_27", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_94", "inbound_nodes": [[["zero_padding2d_84", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_34", "inbound_nodes": [[["conv2d_94", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_85", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_85", "inbound_nodes": [[["max_pooling2d_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_95", "inbound_nodes": [[["zero_padding2d_85", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_35", "inbound_nodes": [[["conv2d_95", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_86", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_86", "inbound_nodes": [[["max_pooling2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_96", "inbound_nodes": [[["zero_padding2d_86", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_34", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_34", "inbound_nodes": [[["conv2d_96", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_87", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_87", "inbound_nodes": [[["up_sampling2d_34", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_97", "inbound_nodes": [[["zero_padding2d_87", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_35", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_35", "inbound_nodes": [[["conv2d_97", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_88", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_88", "inbound_nodes": [[["up_sampling2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_98", "inbound_nodes": [[["zero_padding2d_88", 0, 0, {}]]]}], "input_layers": [["input_27", 0, 0]], "output_layers": [["conv2d_98", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_27", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_27"}}
�
regularization_losses
	variables
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_84", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_84", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_94", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 3]}}
�
 regularization_losses
!	variables
"trainable_variables
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
$regularization_losses
%	variables
&trainable_variables
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_85", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_85", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 3]}}
�
.regularization_losses
/	variables
0trainable_variables
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
2regularization_losses
3	variables
4trainable_variables
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_86", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_96", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 12]}}
�
<regularization_losses
=	variables
>trainable_variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "UpSampling2D", "name": "up_sampling2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_34", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_87", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_87", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_97", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 12, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 16]}}
�
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "UpSampling2D", "name": "up_sampling2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_35", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_88", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_88", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_98", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 12]}}
�
Xiter
	Ydecay
Zlearning_rate
[momentum
\rho
rms�
rms�
(rms�
)rms�
6rms�
7rms�
Drms�
Erms�
Rrms�
Srms�"
	optimizer
 "
trackable_list_wrapper
f
0
1
(2
)3
64
75
D6
E7
R8
S9"
trackable_list_wrapper
f
0
1
(2
)3
64
75
D6
E7
R8
S9"
trackable_list_wrapper
�
regularization_losses
]layer_regularization_losses
^non_trainable_variables
	variables
_metrics

`layers
trainable_variables
alayer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
blayer_regularization_losses
cnon_trainable_variables
	variables
dmetrics

elayers
trainable_variables
flayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_94/kernel
:2conv2d_94/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
glayer_regularization_losses
hnon_trainable_variables
	variables
imetrics

jlayers
trainable_variables
klayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 regularization_losses
llayer_regularization_losses
mnon_trainable_variables
!	variables
nmetrics

olayers
"trainable_variables
player_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
$regularization_losses
qlayer_regularization_losses
rnon_trainable_variables
%	variables
smetrics

tlayers
&trainable_variables
ulayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_95/kernel
:2conv2d_95/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
*regularization_losses
vlayer_regularization_losses
wnon_trainable_variables
+	variables
xmetrics

ylayers
,trainable_variables
zlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
.regularization_losses
{layer_regularization_losses
|non_trainable_variables
/	variables
}metrics

~layers
0trainable_variables
layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
2regularization_losses
 �layer_regularization_losses
�non_trainable_variables
3	variables
�metrics
�layers
4trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_96/kernel
:2conv2d_96/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
�
8regularization_losses
 �layer_regularization_losses
�non_trainable_variables
9	variables
�metrics
�layers
:trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
<regularization_losses
 �layer_regularization_losses
�non_trainable_variables
=	variables
�metrics
�layers
>trainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
@regularization_losses
 �layer_regularization_losses
�non_trainable_variables
A	variables
�metrics
�layers
Btrainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_97/kernel
:2conv2d_97/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
�
Fregularization_losses
 �layer_regularization_losses
�non_trainable_variables
G	variables
�metrics
�layers
Htrainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jregularization_losses
 �layer_regularization_losses
�non_trainable_variables
K	variables
�metrics
�layers
Ltrainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nregularization_losses
 �layer_regularization_losses
�non_trainable_variables
O	variables
�metrics
�layers
Ptrainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_98/kernel
:2conv2d_98/bias
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
�
Tregularization_losses
 �layer_regularization_losses
�non_trainable_variables
U	variables
�metrics
�layers
Vtrainable_variables
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
0
�0
�1"
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
14"
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

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
4:22RMSprop/conv2d_94/kernel/rms
&:$2RMSprop/conv2d_94/bias/rms
4:22RMSprop/conv2d_95/kernel/rms
&:$2RMSprop/conv2d_95/bias/rms
4:22RMSprop/conv2d_96/kernel/rms
&:$2RMSprop/conv2d_96/bias/rms
4:22RMSprop/conv2d_97/kernel/rms
&:$2RMSprop/conv2d_97/bias/rms
4:22RMSprop/conv2d_98/kernel/rms
&:$2RMSprop/conv2d_98/bias/rms
�2�
.__inference_functional_53_layer_call_fn_144457
.__inference_functional_53_layer_call_fn_144520
.__inference_functional_53_layer_call_fn_144739
.__inference_functional_53_layer_call_fn_144714�
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
I__inference_functional_53_layer_call_and_return_conditional_losses_144355
I__inference_functional_53_layer_call_and_return_conditional_losses_144689
I__inference_functional_53_layer_call_and_return_conditional_losses_144393
I__inference_functional_53_layer_call_and_return_conditional_losses_144622�
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
!__inference__wrapped_model_144079�
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
input_27���������  
�2�
2__inference_zero_padding2d_84_layer_call_fn_144092�
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
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_144086�
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
*__inference_conv2d_94_layer_call_fn_144759�
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
E__inference_conv2d_94_layer_call_and_return_conditional_losses_144750�
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
1__inference_max_pooling2d_34_layer_call_fn_144104�
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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_144098�
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
2__inference_zero_padding2d_85_layer_call_fn_144117�
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
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_144111�
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
*__inference_conv2d_95_layer_call_fn_144779�
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
E__inference_conv2d_95_layer_call_and_return_conditional_losses_144770�
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
1__inference_max_pooling2d_35_layer_call_fn_144129�
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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_144123�
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
2__inference_zero_padding2d_86_layer_call_fn_144142�
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
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_144136�
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
*__inference_conv2d_96_layer_call_fn_144799�
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
E__inference_conv2d_96_layer_call_and_return_conditional_losses_144790�
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
1__inference_up_sampling2d_34_layer_call_fn_144161�
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
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_144155�
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
2__inference_zero_padding2d_87_layer_call_fn_144174�
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
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_144168�
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
*__inference_conv2d_97_layer_call_fn_144819�
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
E__inference_conv2d_97_layer_call_and_return_conditional_losses_144810�
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
1__inference_up_sampling2d_35_layer_call_fn_144193�
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
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_144187�
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
2__inference_zero_padding2d_88_layer_call_fn_144206�
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
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_144200�
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
*__inference_conv2d_98_layer_call_fn_144839�
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
E__inference_conv2d_98_layer_call_and_return_conditional_losses_144830�
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
4B2
$__inference_signature_wrapper_144555input_27�
!__inference__wrapped_model_144079�
()67DERS9�6
/�,
*�'
input_27���������  
� "=�:
8
	conv2d_98+�(
	conv2d_98���������  �
E__inference_conv2d_94_layer_call_and_return_conditional_losses_144750l7�4
-�*
(�%
inputs���������""
� "-�*
#� 
0���������  
� �
*__inference_conv2d_94_layer_call_fn_144759_7�4
-�*
(�%
inputs���������""
� " ����������  �
E__inference_conv2d_95_layer_call_and_return_conditional_losses_144770l()7�4
-�*
(�%
inputs���������""
� "-�*
#� 
0���������  
� �
*__inference_conv2d_95_layer_call_fn_144779_()7�4
-�*
(�%
inputs���������""
� " ����������  �
E__inference_conv2d_96_layer_call_and_return_conditional_losses_144790l677�4
-�*
(�%
inputs���������""
� "-�*
#� 
0���������  
� �
*__inference_conv2d_96_layer_call_fn_144799_677�4
-�*
(�%
inputs���������""
� " ����������  �
E__inference_conv2d_97_layer_call_and_return_conditional_losses_144810�DEI�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
*__inference_conv2d_97_layer_call_fn_144819�DEI�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
E__inference_conv2d_98_layer_call_and_return_conditional_losses_144830�RSI�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
*__inference_conv2d_98_layer_call_fn_144839�RSI�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
I__inference_functional_53_layer_call_and_return_conditional_losses_144355�
()67DERSA�>
7�4
*�'
input_27���������  
p

 
� "?�<
5�2
0+���������������������������
� �
I__inference_functional_53_layer_call_and_return_conditional_losses_144393�
()67DERSA�>
7�4
*�'
input_27���������  
p 

 
� "?�<
5�2
0+���������������������������
� �
I__inference_functional_53_layer_call_and_return_conditional_losses_144622|
()67DERS?�<
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
I__inference_functional_53_layer_call_and_return_conditional_losses_144689|
()67DERS?�<
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
.__inference_functional_53_layer_call_fn_144457�
()67DERSA�>
7�4
*�'
input_27���������  
p

 
� "2�/+����������������������������
.__inference_functional_53_layer_call_fn_144520�
()67DERSA�>
7�4
*�'
input_27���������  
p 

 
� "2�/+����������������������������
.__inference_functional_53_layer_call_fn_144714�
()67DERS?�<
5�2
(�%
inputs���������  
p

 
� "2�/+����������������������������
.__inference_functional_53_layer_call_fn_144739�
()67DERS?�<
5�2
(�%
inputs���������  
p 

 
� "2�/+����������������������������
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_144098�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_34_layer_call_fn_144104�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_144123�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_35_layer_call_fn_144129�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
$__inference_signature_wrapper_144555�
()67DERSE�B
� 
;�8
6
input_27*�'
input_27���������  "=�:
8
	conv2d_98+�(
	conv2d_98���������  �
L__inference_up_sampling2d_34_layer_call_and_return_conditional_losses_144155�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_up_sampling2d_34_layer_call_fn_144161�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_up_sampling2d_35_layer_call_and_return_conditional_losses_144187�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_up_sampling2d_35_layer_call_fn_144193�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_zero_padding2d_84_layer_call_and_return_conditional_losses_144086�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_zero_padding2d_84_layer_call_fn_144092�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_zero_padding2d_85_layer_call_and_return_conditional_losses_144111�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_zero_padding2d_85_layer_call_fn_144117�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_zero_padding2d_86_layer_call_and_return_conditional_losses_144136�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_zero_padding2d_86_layer_call_fn_144142�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_zero_padding2d_87_layer_call_and_return_conditional_losses_144168�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_zero_padding2d_87_layer_call_fn_144174�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_zero_padding2d_88_layer_call_and_return_conditional_losses_144200�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_zero_padding2d_88_layer_call_fn_144206�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������