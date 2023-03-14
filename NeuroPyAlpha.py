
"""

Author: Mark John Velmonte
Date: February 2, 2023

Description: Contains class to create simple but expandable neural network from scratch.

"""


from random import uniform
from math import log, sqrt
from VectorMethods import Vector
from MatrixMethods import Matrix, Tensor3D
import pprint
import json
try:
	from tqdm import tqdm
except:
	raise Exception("Missing required library tqdm: please install tqdm via 'pip install tqdm'")


VERSION_CODE = "NeuroPy Version Alpha-0.0.12"


pp = pprint.PrettyPrinter(width = 41, compact = True)






class WeightInitializer():
	def __init__(self):
		"""
			This class contains different methods to generate weights tot thee neural network

			Arguments: takes 0 arguments
		"""
		super().__init__()


	def intializeWeight(self, dim, min_f = -1.0, max_f = 1.0):
		"""
			This method generate weights using simple random number calculations

			Arguments: 
			dim (lsit)		: 	A two lenght list contains the row and columnn [row, col] or shape of the generated weight
			min_f (float) 	:	The minimum value limit
			max_f (float)	:	The maximum value limit
			

			Returns:Vector
		"""
		final_weight_arr = []
		row = dim[0]
		col = dim[1]

		for i in range(row):
			col_vector = []
			for j in range(col):
				rng = 2 * uniform(min_f, max_f) - 1
				col_vector.append(rng)

			final_weight_arr.append(col_vector)
		return Vector(final_weight_arr)


	def initNormalizedXavierWeight(self, dim, n_of_preceding_nodes, n_of_proceding_node):
		"""
			This method generate weights using xavier weight initialization method

			Arguments: 
			dim (list)		: 	A two lenght list contains the row and columnn [row, col] or shape of the generated weight
			n_of_preceding_nodes (Vector)	:	The number of neurons where outputs will come from
			n_of_proceding_node (Vector)		:	The number of neurons that will accepts the outputs frrom the preceding neuro

			Returns:Vector
		"""
		final_weight_arr = []
		row = dim[0]
		col = dim[1]

		for row_count in range(row):

			n = n_of_preceding_nodes
			m = n_of_proceding_node
			sum_of_node_count = n + m
			lower_range, upper_range = -(sqrt(6.0) / sqrt(sum_of_node_count)), (sqrt(6.0) / sqrt(sum_of_node_count))
			rand_num = Vector([uniform(0, 1) for i in range(col)])
			scaled = rand_num.addScalar(lower_range).multiplyScalar((upper_range - lower_range))

			final_weight_arr.append(scaled)

		return Vector(final_weight_arr)
















class ActivationFunction():
	def __init__(self):
		"""
			This class contains different methods that calculate different deep learning functions

			Arguments: takes 0 arguments
		"""
		self.E = 2.71

	def sigmoidFunction(self, x):
		"""
			This method perform a sigmoid function calculation

			Arguments: 
			x(float) 	: The value where sigmoid function will be applied
			

			Returns: float
		"""
		x = x
		result = 1 / (1 + self.E ** -x)

		return result


	def argMax(self, ouput_vector):
		"""
			This method search for the maximum value and create a new list where only the maximum value will have a value of 1

			Arguments: 
			weight_matrix (Matrix) 	: The array that will be transformed into a new array
			
			Returns: Vector
		"""
		output_array = []
		max_value_index = ouput_vector.index(max(ouput_vector))

		for index in range(len(ouput_vector)):
			if index == max_value_index:
				output_array.append(1)
			elif index != max_value_index:
				output_array.append(0)

		return Vector(output_array)



	def relu(self, input_value):
		"""
			Returns the exact input if that input is a positive value
			
			input_value (Scalar / float) 		:  value from the weighted sum of the input
			return (Scalar / float)
		"""
		return max(0, input_value)















class ForwardPropagation(ActivationFunction):
	def __init__(self):
		"""
			This class contains different methods for neural network forward propagation
			Arguments: takes 0 arguments
		"""
		super().__init__()


	def layerForwardPass(self, input_matrix, weight_matrix, bias_weight_matrix, activation_function):
		"""
			Creates a nueral network layer
			Arguments: 
			input_value (matrix) 	: 	testing inputs 
			weight_value (matrix)	:	The corresponding weight to this layer
			bias_weight (matrix)		:	The weight of the bias for this layer
			
			Returns (Vector) : The ouput of the this layer
		"""
		weighted_sum_matrix_set = self.getWeightedSum(weight_matrix, input_matrix)
		biased_weighted_sum_set = self.applyBias(weighted_sum_matrix_set, bias_weight_matrix)

		output_matrix = []
		for weighted_sum  in weighted_sum_matrix_set:
			act_func_vector = []

			for input_val in weighted_sum:
				if activation_function == "sigmoid":
					act_func_vector.append(self.sigmoidFunction(input_val))
				elif activation_function == "relu":
					act_func_vector.append(self.relu(input_val))

			output_matrix.append(act_func_vector)

		return output_matrix


	def getWeightedSum(self, weight_matrix, input_matrix):
		"""
			Caculate weighted sum of the incoming input
			Arguments: 
			input_arr (matrix) 	: 	Inputs eigther from a layer ouput or the main testing data
			weight_arr (matrix)	: 	The generated weight
			Returns (matrix) : Weighted sum 
		"""
		return Matrix().matrixDotProd(input_matrix, weight_matrix)



	def applyBias(self, weighted_sum_matrix_set, bias_weight_matrix):
		"""
			apply the bias to the incoming inputs to the recieving neurons layer

			Arguments: 
			bias_weight_arr (Vector) 		: 	weights of the bias to be added to the incoming inputs
			weighted_sum_arr (Vector)		: 	The generated weight
			
			Returns (Vector) : biased inputs
		"""
		return Matrix().matrixAddition(weighted_sum_matrix_set, bias_weight_matrix)

























class WeightUpdates():
	def __init__(self):
		super().__init__()


	def sigmoidWeightCalculation(self, alpha, fwd_l_delta, prev_l_output_matrix, init_weight):
		""" 
			Calculate and return an array of floats that is intended to use for calibrating the weights of the 
			Neural network
			
			Arguments:
			alpha (scalar / float)				:	Network learning rate
			fwd_l_delta (Vector)				:	The layer infront of the current layer [l + 1]
			prev_l_output (Vector)				:	The layer before the current layer  [l - 1]
			init_weight (matrix)				:	The initial weight without the adjustments

			Returns: (Matrix) Returns the updated weight of the given initial weight value

			formula:
			weight_ajustments = -learning_rate * [outerProduct(fwd_l_delta, prev_l_output)]
		"""


		weight_adjustment_matrix = []

		for prev_l_output in prev_l_output_matrix:
			neighbor_neuron_dprod = Matrix().outerProduct(fwd_l_delta, prev_l_output)
			weight_set = []
			for selected_row in neighbor_neuron_dprod:
				result_row = []

				for col_val in selected_row:
					product = alpha * col_val
					result_row.append(product)

				weight_set.append(result_row)

			weight_adjustment_matrix.append(weight_set)

		return weight_adjustment_matrix



	def sigmoidL2RegularizationWeightUpdate(self, learning_rate, delta_n, prev_layer_output_matrix, l2_lambda, intial_weight):
		"""
			Apply the l2 regularization when calculating the weights of the layers

			Aguemtns:
			learning_rate (float)			: The models learning rate
			delta_n (Vector)				: The strenght of the hidden layer recieving from the weihts being update
			prev_layer_output (Matrix)		: The output of the sactivation function of the previos layer
			l2_lambda (float) 				: L2 penalty
			intial_weight (matrix)			: The initial weight

			return matrix with values representing the amount to change the weight
			Equation: delta_w = alpha * (delta * X.T) + lambda * W
 
		"""
		# HERE IS THE PROBLE
		delta_w_matrix = []
		for output_matrix in prev_layer_output_matrix:
			for prev_layer_output in output_matrix:
				delta_w = Matrix().matrixAddition(
								Vector(delta_n).multiplyVector(Vector(prev_layer_output).multiplyScalar(learning_rate)), 
								Matrix().matrixScalarMultiply(intial_weight, l2_lambda)
								)
				delta_w_matrix.append(delta_w)

		delta_w_vector = Matrix().columnAverage(delta_w_matrix)[0]
		

		# W = W - (alpha * dW + lambda * W)
		weight_matrix_update = Matrix().matrixSubtract(
								intial_weight,
								Matrix().matrixAddition(
									Matrix().matrixScalarMultiply(delta_w_vector, learning_rate), 
									Matrix().matrixScalarMultiply(intial_weight, l2_lambda)
									)
								) 

		return Vector(weight_matrix_update)






	def reluWeightCalculations(self, learning_rate, fwd_l_delta, prev_l_output, init_weight):
		"""
			calculate and update the weights without any regularization
				New:	
						dW = prev_l_output * fwd_l_delta.T
						W = W - learning_rate * (dW + lambda * W) with Lw
						W = W - learning_rate * dW  # Without L2
		"""
		
		
		#weight_adjustment = self.matrixScalarMultiply(self.outerProduct(fwd_l_delta, prev_l_output), learning_rate)
		#weight_update = self.matrixSubtract(init_weight, self.matrixScalarMultiply(weight_adjustment, learning_rate)) 
		
		# NEW CHANGES
		weight_adjustment = Matrix().transpose(Matrix().outerProduct(prev_l_output, fwd_l_delta))
		weight_update = Matrix().matrixAddition(init_weight, Matrix().matrixScalarMultiply(weight_adjustment, learning_rate)) 
						#self.
		return weight_update




	def applyWeightAdjustment(self, initial_weight, weight_adjustment, operation = "+"):
		"""
			Apply the adjustments of the weights to the initial weight to update its value by getting the sum of the two array

			Arguments:
			initial_weight (List / Vector)			:	The weights value that is used in forward propagation
			weight_adjustment  (List / Vector)		:	The value used to add to the initial weight

			Returns: Vector
		"""
		if operation == "+":
			returned_value = Matrix().matrixAddition(initial_weight, weight_adjustment)
		elif operation == "-":
			returned_value = Matrix().matrixSubtract(initial_weight, weight_adjustment)

		return Vector(returned_value)





























class DeltaCalculationMethods():
	def __init__(self):
		super().__init__()


	# rename 
	# preceding_neuron_output_vector to l_output
	# weight_matrix 
	# proceding_neuron_output_matrix to fwd_l_delta


	def sigmoidDeltaCalculation(self, prev_l_output_matrix, weight_matrix, fwd_l_delta):
		"""
			Calculate the delta of the a layer using sigmoid derivative

			Arguments: 
				output_vector_ind (vector) 				:	The output of the layer or the ouput of the sigmoid function in that layer
				weight_matrix (matrix)			:	Updated weight matrix next to the current layer being update
				proceding_neuron_output_matrix (matrix)  (ith_layer + 1) 

			Return (Vertor) Returns the calculated delta of the layer

			Equation : transpose(weight_matrix) matx_mult(fwd_l_delta) * (output_vector_ind (1 - output_vector_ind)))
		"""

		calculated_delta_matrix = []
		for output_vector_ind in prev_l_output_matrix: # FLAG : 
			transposed_weight = Matrix().transpose(weight_matrix)

			subtracted_arr = []
			for neuron_val in output_vector_ind:
				subtracted_arr.append(1 - neuron_val)

			product_arr = []
			for index in range(len(output_vector_ind)):
				product_arr.append(output_vector_ind[index] * subtracted_arr[index])

			dot_product_arr = Matrix().matrixVectorMultiply(transposed_weight, fwd_l_delta)
			sum_of_rows_arr = Matrix().getSumOfRow(dot_product_arr)
			#neuron_delta = Matrix().vectorMultiply(Matrix().flatten(sum_of_rows_arr), product_arr)
			neuron_delta = Vector(Matrix().flatten(sum_of_rows_arr)).multiplyVector(product_arr)

			calculated_delta_matrix.append(neuron_delta)

		
		return Matrix().columnAverage(calculated_delta_matrix)[0]



	def reluDeltaCalculation(self, l_fl_weight_matrix, fwd_l_delta, l_output):
		"""
			Calculate the delta of a layer using the dewrivative of the relu activation function

			Arguments:
			l_fl_weight_matrix (matrix) 		:	The weight matrix in the middle of the current layer and the next layer
			fwd_l_delta (vecctor)				:	The calculated delta of the layer in front of the current layer
			l_output (Vector)					: 	The ouput of the current layer

			Return: (Vector)
			Equation : 
			1. delta_j = dot(l_fl_weight_matrix, fwd_l_delta) * (l_output > 0)
			2. delta_j = relu_derivative(z_j) * sum(w_jk * delta_k)
		"""

		# NEW CHANGES
		relu_derivative = []
		for value in l_output:
			if value > 0:
				relu_derivative.append(1)
			elif value <= 0:
				relu_derivative.append(0)

		weight_delta_sum = Matrix().matrixSum(Matrix().matrixVectorMultiply(Matrix().transpose(l_fl_weight_matrix), fwd_l_delta))
		delta_vector = Vector(relu_derivative).multiplyScalar(weight_delta_sum)
		
		return delta_vector























# HERE 
class BackPropagation(WeightUpdates, DeltaCalculationMethods):
	def __init__(self, learning_rate = -0.01):
		"""
			This class handles the backpropagation acalculation methods
		"""
		super().__init__()
		self.learning_rate = learning_rate



	def getFinalLayerDelta(self, fl_p_out_tensor, actual_label_matrix, activation_function):
		"""
			Calculate the final layer neuron strenghts

			Arguments:
			fl_p_out_tensor (matrix)				:	Final output that is calculated by sigmoid function
			actual_label_matrix (List / Vector)	:	The final ouput that is produced by argmax function

			Returns: Vector
		"""

		output_vector = []

		if activation_function == "sigmoid":
			delta_matrix = []
			for p_vect_index in range(len(fl_p_out_tensor)):
				delta_matrix.append(Vector(fl_p_out_tensor[p_vect_index]).subtractVector(actual_label_matrix[p_vect_index]))

			output_vector = Matrix().columnAverage(delta_matrix)[0]



		elif activation_function == "relu":
			relu_derivative_vector = []
			for p_vector_index in range(len(fl_p_out_tensor)):
				for value in fl_p_out_tensor[p_vector_index]:
					## relu derivative
					if value > 0:
						relu_derivative_vector.append(1)
					elif value <= 0:
						relu_derivative_vector.append(0)

				output_vector = Vector(fl_p_out_tensor[p_vector_index]).subtractVector(actual_label_matrix[p_vector_index]).multiplyVector(relu_derivative_vector)

		else:
			err_msg = "No activation function found : " + activation_function
			raise ValueError(err_msg)


		return Vector(output_vector)



	def getCrossEntropyLoss(self, predicted_ouput_matrix, actual_label_matrix, activation_function):
		"""
			This method is made to calculate the coss entropy loss for the final layer

			Arguments:
			predicted_ouputs_matrix (Vector) (p)		:	Networks final layer or prediction
			actual_label_matrix (Vector) (y)	:	The actual label for the given problem

			Return (Vector) calculated loss

			Equation:
				Softamx : CE = -∑(y * log(softmax(z)))
				Sigmoid: CE = - ∑(y * log(sigmoid(z)) + (1-y) * log(1-sigmoid(z)))
				batch : CE = -1/m * sum(sum(y * log(y_hat) + (1-y) * log(1-y_hat)))
				Where:
					z = predicted probability distribution
		"""
		for predicted_output_vector, actual_label_vector in zip(predicted_ouput_matrix, actual_label_matrix):
			output_vector = []
			for p, y in zip(predicted_output_vector, actual_label_vector):
				if activation_function == "softmax":
					cross_entropy_calculation = y * log(p)
				elif activation_function == "sigmoid":
					cross_entropy_calculation = y * log(p) + (1 - y) * log(1 - p)

				output_vector.append(cross_entropy_calculation)

		return Vector(output_vector).sum()




	

	def updateLayerWeight(self, alpha, fwd_l_delta, prev_l_output, init_weight, activation_function_method, get_updated_weight = True):
		
		"""
			Update weight matrix
			Arguemnts:
			get_weight_update (Boolean)		: If True the function will return the updated weight matrix, If False will return the Update matrix instead of the updated weight 
		"""
		
		if activation_function_method == "sigmoid":
			if len(fwd_l_delta) != 0 and len(prev_l_output) != 0 and len(init_weight) != 0:
				weight_adjustment_matrix = self.sigmoidWeightCalculation(
											alpha = alpha, 
											fwd_l_delta = fwd_l_delta, 
											prev_l_output_matrix = prev_l_output, 
											init_weight = init_weight
											)

				delta_w_vector = Tensor3D().columnAverage(weight_adjustment_matrix)
				weight_update_matrix = self.applyWeightAdjustment(
						initial_weight = init_weight, 
						weight_adjustment = delta_w_vector, 
						operation = "+"
					)

		elif activation_function_method == "relu":
			if alpha != 0 and len(fwd_l_delta) != 0 and len(prev_l_output) != 0 and len(init_weight) != 0:
				weight_update_matrix = self.reluWeightCalculations(
											learning_rate = alpha, 
											fwd_l_delta = fwd_l_delta, 
											prev_l_output = prev_l_output, 
											init_weight = init_weight
											)

		if Matrix().getShape(weight_update_matrix) != Matrix().getShape(init_weight):
			err_msg = "Calculated update weight is not equal to the initial weight with shape " + str(Matrix().getShape(weight_update_matrix)) + " Rather than " + str(Matrix().getShape(init_weight))

			raise Exception(err_msg + "\n With delta_w_vector of >>> " + str(delta_w_vector), "\n weight_update_matrix >>> " + str(weight_update_matrix))

		if get_updated_weight == True:
			return Vector(weight_update_matrix)
		elif get_updated_weight == False:
			return Vector(weight_adjustment_matrix)
		else:
			raise Exception("No weight matrix is returned is returned")





	# FLAG : BROKEN
	def L2regularizedWeightUpdate(self, learning_rate, delta_n, prev_layer_output_matrix, l2_lambda, intial_weight, activation_function = "sigmoid"):
		"""
			Update weight matrix with a regularization method

		"""
		weight_update_matrix = []
		if activation_function == "sigmoid":
			calculated_weight_matrix = self.sigmoidL2RegularizationWeightUpdate(learning_rate, delta_n, prev_layer_output_matrix, l2_lambda, intial_weight)
	
		return weight_update_matrix



	def getHiddenLayerDelta(self, prev_l_output_matrix, weight, fwd_l_delta, activation_function):
		"""
			calculate the strenght of the neurons in a hidden layer 
			
			Arguments:
			prev_l_output (List / Vector)				:	The layer of neurons that is first to recieve data relative to forward propagation direction
			weights (List / Vector) 						:	The weights in the middle to the two given neurons
			proceding_neuron_strenght (List / Vector)	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			
			Retuns: Vector

		"""
		if activation_function == "sigmoid":
			delta_vector = self.sigmoidDeltaCalculation(
								prev_l_output_matrix = prev_l_output_matrix, 
								weight_matrix = weight, 
								fwd_l_delta = fwd_l_delta
								)

		elif activation_function == "relu":
			delta_vector = self.reluDeltaCalculation(
								l_fl_weight_matrix = weight, 
								fwd_l_delta = fwd_l_delta, 
								l_output = prev_l_output_matrix
							)
		else:
			err_msg = "No Activation function named: " + activation_function
			raise ValueError(err_msg)

		return delta_vector




	def adjustBiasWeight(self, l_delta, init_bias, learning_rate, activation_function):
		"""
			Calculate bias adjustment
			
			Argumemts:
			l_delta	(List / Vector)	:	Updated neuron delta

			Formula: 
				ReLu 			:		delta_w_bias = learning_rate * error_signal * 1
										new_bias = current_bias + delta_w_bias
				sigmoid 		:		-learning_rate * updated_neuron_strenght
			
			Return Vector
		"""

		if  activation_function == "relu":
			delta_bias = Vector(l_delta).multiplyScalar(learning_rate).multiplyScalar(1)
			adjusted_biase = [Vector(init_bias[0]).addVector(delta_bias)]

		elif activation_function == "sigmoid":
			adjusted_biase = [Vector(l_delta).multiplyScalar(self.learning_rate)]

		return Vector(adjusted_biase)
		


	def getMeanSquaredError(self, predicted_ouputs_vector, actual_label_vector):
		"""
			Calculate the mean squared error or cost value

			Arguments;
			predicted_ouputs_vector (List / Vector) 				:	The unlabled output, or the output from the sigmoid function
			actual_label_vector (List / Vector)		:	The labled output
			
			returns : float
			Formula : 1 / lne(predicted_ouputs_vector) * sum((predicted_ouputs_vector - actual_label_vector) ** 2)

		"""

		#arr_difference = self.vectorSubtract(predicted_ouputs_vector, actual_label_vector)
		#squared_arr = self.vectorSquare(arr_difference)

		arr_difference = Vector(predicted_ouputs_vector).subtractVector(actual_label_vector)
		squared_arr = Vector(arr_difference).squared()
 
		arr_sum = Vector(squared_arr).sum()
		e = 1 / 3 * arr_sum

		return e



	def sgdMiniBatchWeightUpdate(self, init_weight, layer_weight_updates_matrix, learning_rate, mini_batch_size):
		"""
			Apply the stochastic gradient decent by mini-batch
			
			Arguments:
			init_weight (Matrix)								: The initial weight of the layer
			layer_weight_updates_matrix (multidim Matrix)		: Contains the n weight update matrix where n is the number of the mini batch

			Equation:
			W = W - (learning_rate / m) * sum(grads)
				Where:
				W = is the weight matrix of the network, 
				learning_rate = is the step size or learning rate hyperparameter, 
				m = is the batch size
				grads = is the average gradient of the loss with respect to the weights across all examples in the batch.

			return (Matrix) updated weight
		"""

		grad = self.matrixSum(self.matrixAverage(layer_weight_updates_matrix))
		trm_2 = (learning_rate / mini_batch_size) * grad
		final_weight_update = self.matrixScalarSubtract(init_weight, trm_2)

		return final_weight_update





























class CreateNetwork(ForwardPropagation, BackPropagation):
	def __init__(self, input_size, layer_size_vectors, learning_rate = -0.01, weight_initializer = "xavierweight", regularization_method = "none", l2_penalty = 0.01):
		super().__init__()
		self.learning_rate = learning_rate
		self.input_size = input_size
		self.layer_size_vectors = layer_size_vectors
		self.weight_initializer = weight_initializer

		self.l2_penalty = l2_penalty
		self.regularization_method = regularization_method

		self.layer_sizes = self.initailizeLayerSizes()
		self.weights_set = self.initializeLayerWeights()
		self.bias_weight_set = self.initializeBiasedWeights()
		self.mean_square_error_log = []
		self.batch_array = []
		self.answer_key_batch_array = []
		self.accuracy = 0.0



	def fit(self, training_data, labeld_outputs, epoch, batch_size):
			"""
				Arguments: 
				training_data (Matrix)			: Matrix of the training data
				labeld_outputs (Matrix)			: Matrix of the labled output of the training data
				epoch (scalar int)				: The amount of loop i will do to look over the entire training data
				batch_size (scalar int)			: The amount of batches of training data to be trained

			"""
			self.printNetworkPrelimSummary(epoch, batch_size)
			
			# Devide the training data in batches
			self.batch_array, self.answer_key_batch_array = self.devideBatches(training_data, labeld_outputs, batch_size)

			# get the value of hopw many layers the network have
			layer_count = len(self.layer_sizes)

			# get how many batches is needed to loop through
			batches_count = len(self.batch_array)

			# count the number of correct prediction to calculate accuracy of the model
			correct_prediction = 0


			for _ in tqdm(range(epoch)):
				for training_batch_set, labeld_batch_key in zip(self.batch_array, self.answer_key_batch_array):

					# get the input data for the current loop
					input_data = training_batch_set
					# get the labeld output of the input data
					batch_input_labeld_data = labeld_batch_key


					#### FORWARD PROPAGATION ####
					# list every output of the neuron for every layer
					batch_layer_ouput_tensor = []

					# The output of the previous layer and the input of the current layer in iteration
					current_layer_input = input_data

					# Loop through the entire layer of the neural network for forward pass
					for layer_index in range(layer_count):
						layer_activation_function = self.layer_size_vectors[layer_index][1]
						
						# create a layer where neuron activation and other transformation will handle
						current_layer_input = self.layerForwardPass(
											input_matrix = current_layer_input,
											weight_matrix = self.weights_set[layer_index], 
											bias_weight_matrix = self.bias_weight_set[layer_index],
											activation_function = layer_activation_function
											)

						# Append the output of the layer for backpropagation
						batch_layer_ouput_tensor.append(current_layer_input)
						# update the input for the next layer
						current_layer_input = current_layer_input



					## Check if the model prediction was correct
					for predicted_output, actual_output in zip(batch_layer_ouput_tensor[-1], batch_input_labeld_data):
						final_prediction = ActivationFunction().argMax(predicted_output)
						if final_prediction == actual_output:
							correct_prediction += 1


					self.calculateModelAccuracy(
						n_of_correct_pred = correct_prediction, 
						n_of_training_data = len(training_data),
						training_epoch = epoch
						)

					#### calculate Loss functions ####
					mean_square_error = self.getCrossEntropyLoss(
						 						predicted_ouput_matrix = batch_layer_ouput_tensor[-1],
						 						actual_label_matrix = batch_input_labeld_data,
						 						activation_function = layer_activation_function
						 						)
					
					# Append the result to the error log
					self.mean_square_error_log.append(mean_square_error)

					self.BackPropagationProcess(
						layers_output_tensor = batch_layer_ouput_tensor,
						actual_label_matrix = batch_input_labeld_data,
						layer_input_matrix = input_data
						)

			self.printFittingSummary()



	def BackPropagationProcess(self, layers_output_tensor, actual_label_matrix, layer_input_matrix):
		# 1. calculate the Final layer delta
		try:
			fl_d = self.getFinalLayerDelta(
				fl_p_out_tensor = layers_output_tensor[-1], 
				actual_label_matrix = actual_label_matrix, 
				activation_function = self.layer_size_vectors[-1][1]
	 			)
		except Exception as err:
			err_msg = "Error getting final layer data with a mesage: >>> " + str(err),
			raise Exception()


		# 2-3. Loop through the entire network
		# 2. calculate and update weight to the current layer
		# 3. calculate the delta of the current layer
		fwd_l_d = fl_d # This delta value will be updated for every layer

		for layer_index in range(len(self.layer_sizes) - 1, -1, -1):
			layer_activation_function = self.layer_size_vectors[layer_index][1]
			try:
				self.bias_weight_set[layer_index] = self.adjustBiasWeight(
									l_delta = fwd_l_d,
									init_bias = self.bias_weight_set[layer_index],
									learning_rate = self.learning_rate,
									activation_function = layer_activation_function
									)

				if layer_index != 0: # if the layer is a hidden layer
					l_w_update = self.updateLayerWeight(
						alpha = self.learning_rate, 
						fwd_l_delta = fwd_l_d, 
						prev_l_output = layers_output_tensor[layer_index - 1], 
						init_weight = self.weights_set[layer_index], 
						activation_function_method = layer_activation_function 
						)

				elif layer_index == 0: # if the layer is the input layer
					l_w_update = self.updateLayerWeight(
						alpha = self.learning_rate, 
						fwd_l_delta = fwd_l_d, 
						prev_l_output = layer_input_matrix, 
						init_weight = self.weights_set[layer_index], 
						activation_function_method = "sigmoid" # FLAG : TEMPORARY 
						)


				self.weights_set[layer_index] = l_w_update
				cld_l_d = self.getHiddenLayerDelta(
						prev_l_output_matrix = layers_output_tensor[layer_index - 1],
						weight = l_w_update,
						fwd_l_delta = fwd_l_d,
						activation_function = "sigmoid" # FLAG : TEMPORARY
						)

				fwd_l_d = cld_l_d
				
				
			except Exception as err:
				print("ERROR in index: ", layer_index)
				print("learning_rate >>> ", self.learning_rate, "\n", "fwd_l_d >>> ",fwd_l_d, "\n", "layers_output_tensor >>> ", layers_output_tensor[layer_index - 1], "weights_set[layer_index] >>> ", self.weights_set[layer_index])
				raise Exception(err)









	def predict(self, input_data):
		layer_input = input_data
		layer_output_arr = []

		for layer_index in range(len(self.layer_sizes)):
			# create a layer where neuron activation and other transformation will handle
			layer_ouput = self.layerForwardPass(
							input_value = layer_input, 
							weight_value = self.weights_set[layer_index], 
							bias_weight = self.bias_weight_set[layer_index],
							activation_function = "sigmoid"
							)
			layer_output_arr.append(layer_ouput)
			layer_input = layer_ouput
		
		return layer_output_arr[-1]





	def initailizeLayerSizes(self):
		layer_sizes = []
		for layer_index in range(len(self.layer_size_vectors)):
			current_layer_size = self.layer_size_vectors[layer_index][0]

			if layer_index == 0:
				new_layer = [current_layer_size, self.input_size]
			elif layer_index != 0 :
				new_layer = [current_layer_size, self.layer_size_vectors[layer_index - 1][0]]

			layer_sizes.append(new_layer)
		return layer_sizes



	def initializeLayerWeights(self):
		new_weight_set = []
		for layer_index in range(len(self.layer_sizes)):
			if self.weight_initializer == "xavierweight":
				if layer_index != len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							self.layer_sizes[layer_index], 
							self.layer_sizes[layer_index][0], 
							self.layer_sizes[layer_index + 1][0]
						)
				
				elif layer_index == len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							self.layer_sizes[layer_index], 
							self.layer_sizes[layer_index][0],
							0
						)

			elif self.weight_initializer == "simple":
				new_weight = WeightInitializer().intializeWeight(self.layer_sizes[layer_index])


			new_weight_set.append(new_weight)

		return new_weight_set



	def initializeBiasedWeights(self):
		new_bias_weight_set = []

		for layer_index in range(len(self.layer_sizes)):
			bias_weight_dim = [1, self.layer_sizes[layer_index][0]]

			if self.weight_initializer == "xavierweight":
				if layer_index != len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							bias_weight_dim, 
							self.layer_sizes[layer_index][0], 
							self.layer_sizes[layer_index + 1][0]
						)

				elif layer_index == len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							bias_weight_dim, 
							self.layer_sizes[layer_index][0],
							0
						)


			elif self.weight_initializer == "simple":
				new_weight = WeightInitializer().intializeWeight(bias_weight_dim)

	
			new_bias_weight_set.append(new_weight)

		return new_bias_weight_set



	def devideBatches(self, train_data_arr, answer_key_arr, batch_size):
		test_data_lenght = len(train_data_arr)

		if test_data_lenght < batch_size:
			raise ValueError("Bacth size cannot be grater that the size of the training data")

		test_data_batch_array = []
		answer_key_batch_array = []

		for index in range(0, test_data_lenght, batch_size):
			test_data_batch_array.append(train_data_arr[index: batch_size + index])
			answer_key_batch_array.append(answer_key_arr[index: batch_size + index])

		return test_data_batch_array, answer_key_batch_array



	def printNetworkPrelimSummary(self, epoch, batch_size):
		tab_distance = 4
		tab = "...." * tab_distance
		print("#" * 34, "Network Summary", "#" * 34)
		print("Fitting model with ", epoch, " epoch and ", batch_size, " Batch size")

		print("\nNetwork Architecture: ")
		print("	Learning Rate:", tab, tab, tab, self.learning_rate)
		print("	Regularization:", tab, tab, tab, self.regularization_method)

		if self.regularization_method == "L2":
			print("	L2-Penalty:		", tab, tab, tab,self.l2_penalty)


		print("Layers: ")
		for _layer_index in range(len(self.layer_sizes)):
			print("	Layer: ", _layer_index + 1,  "	Activation Function: ", self.layer_size_vectors[_layer_index][1], tab, self.layer_sizes[_layer_index][0], " Neurons")

		print("\nFitting Progress:")



	def printFittingSummary(self):
		tab_distance = 4
		tab = "...." * tab_distance

		print("\nTraining Complete: ")
		print("Model accuracy: ", self.accuracy)
		print("#" * 34, "End of Summary", "#" * 34)



	def evaluateBatchOutput(self, prediction_matrix, labeld_matrix):
		correct_prediction = 0

		for p_vect_ind in range(len(prediction_matrix)):
			if prediction_matrix[p_vect_ind] == labeld_matrix[p_vect_ind]:
				correct_prediction += 1

		return correct_prediction



	def calculateModelAccuracy(self, n_of_correct_pred, n_of_training_data, training_epoch):
		"""
			Calculate the model accurary

			Arguments:
				n_of_correct_pred (scala intiger)			: The number of correct prediction the model made
				n_of_training_data (scalar intiger)			: The total number of the trianing data fed during training
 		"""

		self.accuracy = (n_of_correct_pred / (n_of_training_data * training_epoch)) * 100



	def setWeightSetsToLoaded(self, loaded_weight_set, loaded_bias_weight_set):
		self.weights_set = loaded_weight_set
		self.bias_weight_set = loaded_bias_weight_set



	def saveModelToJson(self, fname = "NeuroPyModel"):
		data_to_save = {
		"version_code" : VERSION_CODE,
		"input_size" : self.input_size,
		"layer_size_vectors" : self.layer_size_vectors,
		"learning_rate" : self.learning_rate,
		"weight_initializer" : self.weight_initializer,
		"l2_penalty" : self.l2_penalty,
		"regularization_method" : self.regularization_method,
		"weights_set" : self.weights_set,
		"bias_weight_set" : self.bias_weight_set
		}

		json_obj = json.dumps(data_to_save, indent = 4)

		file_name = fname + ".json"
		with open(file_name, "w") as outfile:
			outfile.write(json_obj)

		print(file_name, " Saved complete ")





class LoadModel(CreateNetwork):
	def __init__(self, file_path):
		self.file_path = file_path
		self.loaded_network_data = self.loadNetworkJsonData()

		super().__init__(
			input_size = self.loaded_network_data["input_size"], 
			layer_size_vectors = self.loaded_network_data["layer_size_vectors"], 
			learning_rate = self.loaded_network_data["learning_rate"], 
			weight_initializer = self.loaded_network_data["weight_initializer"], 
			regularization_method = self.loaded_network_data["regularization_method"], 
			l2_penalty = self.loaded_network_data["l2_penalty"]
			)

		self.setWeightSetsToLoaded(
			loaded_weight_set = self.loaded_network_data["weights_set"], 
			loaded_bias_weight_set = self.loaded_network_data["bias_weight_set"]
			)


	def loadNetworkJsonData(self):
		try:
			with open(self.file_path, 'r') as openfile:
				json_object = json.load(openfile)

				return json_object
		except:
			err_msg = "Failed loading " + self.file_path + " file"
			raise Exception(err_msg)




class DataManager():
	def __init__(self):
		pass


	def trainTestSplit(self, input_data, labeld_data, training_partition= 0.2):
		input_data_lenght = len(input_data)
		labeld_data_lenght = len(labeld_data)

		if input_data_lenght != labeld_data_lenght:
			err_msg = "Shape of Input data is not equal to the labeld data" + str(input_data_lenght) + " != " + str(labeld_data_lenght)
			raise ValueError(err_msg)

		if input_data_lenght < 10 and training_partition < 0.5:
			err_msg = "The dataset is too small to be split with in given training_partition argument: " + str(training_partition)
			raise ValueError(err_msg)


		training_partition = int(training_partition * input_data_lenght)

		trainig_input_data = []
		test_input_data = []

		training_labeld_data = []
		test_labeld_data = []

		last_data_indexed = 0
		for data_index in range(training_partition):
			trainig_input_data.append(input_data[data_index])
			training_labeld_data.append(labeld_data[data_index])

			last_data_indexed += 1

		remaining_data = len(input_data) - last_data_indexed
		for _ in range(remaining_data):
			test_input_data.append(input_data[last_data_indexed])
			test_labeld_data.append(labeld_data[last_data_indexed])

			last_data_indexed += 1

		return trainig_input_data, training_labeld_data, test_input_data, test_labeld_data


