
"""

Author: Mark John Velmonte
Date: February 2, 2023

Description: Contains class to create simple but expandable neural network from scratch.

"""

from random import uniform
from math import log, sqrt
import json
try:
	from tqdm import tqdm
except:
	raise Exception("Missing required library tqdm: please install tqdm via 'pip install tqdm'")


"""
class Math():
	def __init__(self):
		pass

	def getSqrt(self, num, decimal_place = 4):

	    Calculates the square root of a given number

	    Parameters:
	    num (float)			:	The number whose square root needs to be calculated.
	    decimal_place(int)	:	The amount of decimal places before roundoff

	    Returns:
	    float: The square root of the given number.

	    
	    if num < 0:
	        raise ValueError("Error on function square_root: Square root of negative numbers is undefined")
	    elif num == 0:
	        return 0
	    elif num > 0:
	        guess = num / 2.0
	        while True:
	            new_guess = (guess + num / guess) / 2.0
	            if abs(new_guess - guess) < 0.0001:
	                return round(new_guess, decimal_place)

	            guess = new_guess
"""




class ArrayMethods():
	def __init__(self):
		pass


	def matrixVectorDotProd(self, matrix, vector):
		"""
			Calculate the dot product of a vector and a matrix

			Parameters:
				vector (list): A 1-dimensional list representing a vector.
				matrix (list): A 2-dimensional list representing a matrix.

			Returns (Vector) A vector representing the result of the dot product.
		"""
		if len(vector) != len(matrix[0]):
			raise ValueError("The dimensions of the vector and matrix do not match.")

		result = []
		for i in range(len(matrix)):
		    dot_product = 0
		    for j in range(len(vector)):
		        dot_product += vector[j] * matrix[i][j]
		    result.append(dot_product)
		    
		return result


	def vectorDotProduct(self, multiplicand_vector, multiplier_vector):
		"""
			calculate the dot product of the two vector 
			
			Argumemts:
			multiplicand_vector	(Vector)	
			multiplier_vector	(Vector)

			Return (matrix) with the shape row = lenght(multiplicand_vector), col = len(multiplier_vector)
		"""
		output_matrix = []

		for multiplicand_value in multiplicand_vector:
			row_arr = []
			for multiplier_value in multiplier_vector:
				product = multiplicand_value * multiplier_value
				row_arr.append(product)

			output_matrix.append(row_arr)

		return output_matrix



	def matrixVectorMultiply(self, multiplicand_matrix, multiplicand_vecotr):
		"""
			multiply matrix to a vector
			
			Argumemts:
			multiplicand_matrix	(Matrix) 
			multiplicand_vecotr	(Vector)

			Return (Matrix) A two dimensional matrix 
		"""
		output_array = []

		for selected_row in multiplicand_matrix:
			new_row = []
			for index in range(len(selected_row)):
				new_row.append(selected_row[index] * multiplicand_vecotr[index])
			output_array.append(new_row)

		return output_array



	def getMatrixSumOfRow(self, matrix):
		"""
			caculate the sum of rows of a 2d matrix array
			
			Argumemts:
			matrix	(Matrix)

			Return Matrix
		"""
		final_arr = []

		for selected_row in matrix:
			sum_of_current_iter = 0
			for value in selected_row:
				sum_of_current_iter += value

			final_arr.append([sum_of_current_iter])

		return final_arr



	def vectorMultiply(self, vector_array_01, vector_array_02):
		"""
			Multiply two 1d vector
			
			Argumemts:
			vector_array_01	(1d vector Array)
			vector_array_02 (1d vector Array)

			Return vector array
		"""
		array_01_lenght = len(vector_array_01)
		array_02_lenght = len(vector_array_02)

		if array_01_lenght != array_02_lenght:
			raise Exception("Error: Array are not equal where size is ", array_01_lenght, " != ", array_02_lenght)

		output_array = []
		loop_n = int((array_01_lenght + array_02_lenght) / 2)

		for index in range(loop_n):
			output_array.append(vector_array_01[index] * vector_array_02[index])

		return output_array



	def matrixAddition(self, matrix_01, matrix_02):
		"""
			add two matrix
			
			Argumemts:
			matrix_01	(2d matrix Array)
			matrix_02	(2d matrix Array)

			Return 2d Matrix Array
		"""

		if len(matrix_01) != len(matrix_02) or len(matrix_01[0]) != len(matrix_02[0]):
		    raise ValueError("Arrays must have the same shape")

		result = [[0 for _ in range(len(matrix_01[0]))] for _ in range(len(matrix_01))]

		for i in range(len(matrix_01)):
			for j in range(len(matrix_01[0])):
				result[i][j] = matrix_01[i][j] + matrix_02[i][j]

		return result



	def matrixSubtract(self, matrix_minuend, matrix_subtrahend):
		"""
			subtract two matrix
			
			Argumemts:
			matrix_minuend	(2d matrix Array)
			matrix_subtrahend	(2d matrix Array)

			Return 2d Matrix Array
		"""
		if len(matrix_minuend) != len(matrix_subtrahend) or len(matrix_minuend[0]) != len(matrix_subtrahend[0]):
			raise ValueError("Arrays must have the same shape")

		result = [[0 for _ in range(len(matrix_minuend[0]))] for _ in range(len(matrix_minuend))]

		for i in range(len(matrix_minuend)):
			for j in range(len(matrix_minuend[0])):
				result[i][j] = matrix_minuend[i][j] - matrix_subtrahend[i][j]

		return result


	def vectorSubtract(self, minuend_vector_array, subtrahend_vector_array):
		"""
			subtract two matrix
			
			Argumemts:
			minuend_vector_array	(1d vector Array)
			subtrahend_vector_array	(1d vector Array)

			Return 1d vector Array
		"""
		subtracted_arr = []
		minuend_arr_size = len(minuend_vector_array)
		subtrahend_arr_size = len(subtrahend_vector_array)

		if minuend_arr_size != subtrahend_arr_size:
			raise Exception(str("Error on function 'vectorSubtract'. Arrays are not equal lenght with sizes " + str(minuend_arr_size) + " and " + str(subtrahend_arr_size)))

		index_count = int((minuend_arr_size + subtrahend_arr_size) / 2)
		for index in range(index_count):
			subtracted_arr.append(minuend_vector_array[index] - subtrahend_vector_array[index])

		return subtracted_arr


	def flatten(self, matrix_arr):
		"""
			transform a matrix array into a 1d vector array
			
			Argumemts:
			matrix_arr	(nd vector Array)

			Return 1d vector Array
		"""
		flattened = []

		if len(self.getShape(matrix_arr)) == 1:
			return matrix_arr

		for element in matrix_arr:
		    if isinstance(element, list):
		        flattened.extend(self.flatten(element))
		    else:
		        flattened.append(element)

		return flattened


	def getShape(self, array_arg):
		"""
			get the shape of vector / matrix array
			
			Argumemts:
			array_arg	(nd Array)

			Return 1d vector Array
		"""
		shape = []

		while isinstance(array_arg, list):
		    shape.append(len(array_arg))
		    array_arg = array_arg[0]

		return shape


	def transpose(self, arr_arg):
		""""
			Transpose the given 2d matrix array swapping its elemtns positions

			Arguements			:	arr_arg
			Returns(Array) 		:	matrix array
		"""

		if len(self.getShape(arr_arg)) <= 1:
			return arr_arg

		shape = self.getShape(arr_arg)
		transposed_list = [[None]*shape[0] for _ in range(shape[1])]

		for i in range(shape[0]):
		    for j in range(shape[1]):
		        transposed_list[j][i] = arr_arg[i][j]

		return transposed_list


	def vectorSquare(self, vector_array):
		""""
			Transpose the given 2d matrix array swapping its elemtns positions

			Arguements			:	arr_arg
			Returns(Array) 		:	matrix array
		"""
		squared_arr = []
		arr_shape = self.getShape(vector_array)
		if len(arr_shape) != 1:
			raise Exception("Error in function vectorSquare, Array should be one dimesion but have " + str(arr_shape))

		for value in vector_array:
			squared_arr.append(value ** 2)

		return squared_arr


	def vectorScalarMultiply(self, vector_array, multiplier_num):
		""""
			apply scalar multiplication to the given array using the given multiplier

			Arguements:
			vector_array (1d vector array)		:	1d vector array
			multiplier_num (float) 				:	float

			Returns: 1d vector array
		"""
		resulting_arr = []

		for value in vector_array:
			resulting_arr.append(value * multiplier_num)

		return resulting_arr



	# rename from matixScalaMultiply to matrixScalarMultiply
	def matrixScalarMultiply(self, matrix, scalar_multiplier):
		"""
			Multiple a mtrix to a scalar value

			Arguments:
			matrix (matrix)					: 	The matrix that will be multiplied
			scalar_multiplier (Scalar) 		:	The multiplier scalar valuee

			Return (Matrix)
		"""
		output_array = []

		for row in matrix:
			col_val_sum = []
			for col_val in row:
				col_val_sum.append(col_val * scalar_multiplier)

			output_array.append(col_val_sum)

		return Array(output_array)











class Array(list):
	def __init__(self, data):
		"""
			Create new Array object to extend Python list functionality
		"""

		super().__init__(data)
		#self.shape = (len(data),)

		self.shape = self.getShape()


	def transpose(self):
		"""
			Transpose the multidimensional this (Array) object swapping its elemtns positions

			Arguements			:	Takes 0 argumnts
			Returns(Array) 		:	Transposed version of this (Array) object
		"""

		shape = self.getShape()
		if len(self.shape) == 1:
			return Array(self)

		transposed_list = [[None]*shape[0] for _ in range(shape[1])]

		for i in range(shape[0]):
			for j in range(shape[1]):
				transposed_list[j][i] = self[i][j]

		return Array(transposed_list)


	def getShape(self):
		"""
			Get the shape of the this (Array) object

			Arguments 		:	Takes 0 arguments
			Returns(Array)	:	The hape of this (Array) objects shape
		"""
		shape = []
		arr = self


		while isinstance(arr, list):
			shape.append(len(arr))
			arr = arr[0]

		self.shape = shape
		return shape



	def multiply(self, multiplier):
		"""
			Multiply this (Array) object

			Arguments:
			multiplier(float)		:	The number use to multiply each element in this Array

			Return: Array

		"""
		array_product = []

		for value in self:
			array_product.append(value * multiplier)

		return Array(array_product)



	def add(self, addends):
		"""
			Add this (Array) object

			Arguments:
			addends(float)		:	The number use to Add each element in this Array

			Return: Array

		"""
		addends_arr = []

		for value in self:
			addends_arr.append(value + addends)

		return Array(addends_arr)



	def subtract(self, subtrahend):
		"""
			Subtract this (Array) object

			Arguments:
			subtrahend(float)		:	The number use to Subtract each element in this Array

			Return: Array

		"""
		difference = []

		for value in self:
			difference.append(value - subtrahend)

		return Array(difference)


	def sum(self):
		"""
			get the sum of all alements in array

			Arguments: takes 0 arguments
			Return: float
		"""
		total = 0
		for value in self:
			total += value

		return total


	def min(self):
		"""
			get the minimum or lowest value in this (Array) object

			Arguments: takes 0 arguments
			Return: float
		"""
		min_val = 0
		for value in self:
			if value < min_val:
				min_val = value

		return min_val


	def max(self):
		"""
			get the maximum or highest value in this (Array) object

			Arguments: takes 0 arguments
			Return: float
		"""
		max_val = 0
		for value in self:
			if value > max_val:
				max_val = value

		return max_val


	def mean(self):
		"""
			caculkate the mean value of this (Array) object

			Arguments: takes 0 arguments
			Return: float
		"""
		sum_of_arr = self.sum()
		mean_val = sum_of_arr / len(self)

		return mean_val


	def squared(self):
		"""
			caculkate the squared value of every elements in this (Array) object

			Arguments: takes 0 arguments
			Return: Array
		"""
		squared_arr = []
		for value in self:
			squared_arr.append(value ** 2)

		return Array(squared_arr) 


	def std(self):
		"""
			caculate the standard deviation value of this (Array) object

			Arguments: takes 0 arguments
			Return: float
		"""
		#standard_dev = Math().sqrt(self.subtract(self.mean()).squared().sum() / len(self))
		standard_dev = sqrt(self.subtract(self.mean()).squared().sum() / len(self))
		return standard_dev



	def addArray(self, addends_arr):
		"""
			Add two array 

			Arguments:
			addends_arr(Array / List)		:	The array use to add to this array

			Return: Array
		"""
		sum_arry = []

		if self.shape != Array(addends_arr).shape:
			raise Exception("Error on function addArray, Values are not the same shape")

		for index in range(len(self)):
			sum_arry.append(self[index] + addends_arr[index])

		return Array(sum_arry)



	def vectorMultiply(self, multiplier_vector):
		"""
		Return: Matrix
		""" 
		output_vector = []

		for value in self:
			row_vector = []
			for element in multiplier_vector:
				row_vector.append(value * element)

			output_vector.append(row_vector)

		return Array(output_vector)



	def minMaxNorm(self):
		normalized_arr = []

		for value in self:
			norm = (value - self.max()) / (self.max() - self.min())
			normalized_arr.append(norm)

		return normalized_arr








class WeightInitializationMethods():
	def __init__(self):
		"""
			Methods to intializ random value generated using different mathematical functions

			Arguments: Takes 0 arguments
		"""
		#super().__init__()
		pass



	def radomInitializer(self, min_f = 0, max_f = 1.0):
		"""
			Generate random number in range of given paramenter using basic calculation technique

			Arguments: 
			min_f (float) 	:	The minimum value limit
			max_f (float)	:	The maximum value limit

			Returns:float
		"""
		rwg = 2 * uniform(min_f, max_f) - 1

		return rwg


	def NormalizedXavierWeightInitializer(self, col_size, n_of_preceding_nodes, n_of_proceding_node):
		"""
			Generate random number using xavier weight intializer 

			Arguments: 
			col_size (float) 				:	the number of elements or weights to be generated since this will be a 1d array
			n_of_preceding_nodes (Array)	:	The number of neurons where outputs will come from
			n_of_proceding_node (Array)		:	The number of neurons that will accepts the outputs frrom the preceding neuro

			Returns:Array
		"""
		n = n_of_preceding_nodes
		m = n_of_proceding_node

		sum_of_node_count = n + m

		lower_range, upper_range = -(sqrt(6.0) / sqrt(sum_of_node_count)), (sqrt(6.0) / sqrt(sum_of_node_count))
		rand_num = Array([uniform(0, 1) for i in range(col_size)])
		scaled = rand_num.add(lower_range).multiply((upper_range - lower_range))

		return Array(scaled)








class WeightInitializer(WeightInitializationMethods):
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
			

			Returns:Array
		"""
		final_weight_arr = []
		row = dim[0]
		col = dim[1]

		for i in range(row):
			col_arr = []
			for j in range(col):
				col_arr.append(self.radomInitializer(min_f, max_f))

			final_weight_arr.append(col_arr)

		return Array(final_weight_arr)


	def initNormalizedXavierWeight(self, dim, n_of_preceding_nodes, n_of_proceding_node):
		"""
			This method generate weights using xavier weight initialization method

			Arguments: 
			dim (list)		: 	A two lenght list contains the row and columnn [row, col] or shape of the generated weight
			n_of_preceding_nodes (Array)	:	The number of neurons where outputs will come from
			n_of_proceding_node (Array)		:	The number of neurons that will accepts the outputs frrom the preceding neuro

			Returns:Array
		"""

		final_weight_arr = []
		row = dim[0]
		col = dim[1]

		for row_count in range(row):
			data = self.NormalizedXavierWeightInitializer(col, n_of_preceding_nodes, n_of_proceding_node)
			final_weight_arr.append(data)

		return Array(final_weight_arr)









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
			
			Returns: Array
		"""
		output_array = []

		max_value_index = ouput_vector.index(max(ouput_vector))

		for index in range(len(ouput_vector)):
			if index == max_value_index:
				output_array.append(1)
			elif index != max_value_index:
				output_array.append(0)

		return Array(output_array)



	def relu(self, input_value):
		"""
			Returns the exact input if that input is a positive value
			
			input_value (Scalar / float) 		:  value from the weighted sum of the input
			return (Scalar / float)
		"""
		if input_value > 0:
			return input_value
		elif input_value <= 0:
			return 0.0








class ForwardPropagation(ActivationFunction):
	def __init__(self):
		"""
			This class contains different methods for neural network forward propagation

			Arguments: takes 0 arguments
		"""
		super().__init__()


	def fowardPropagation(self, input_value, weight_value, bias_weight, activation_function):
		"""
			Creates a nueral network layer

			Arguments: 
			input_value (Array) 	: 	testing inputs 
			weight_value (Array)	:	The corresponding weight to this layer
			bias_weight (Array)		:	The weight of the bias for this layer
			
			Returns (Array) : The ouput of the this layer
		"""
		weighted_sum = self.getWeightedSum(input_value, weight_value)
		biased_weighted_sum = Array(weighted_sum).addArray(ArrayMethods().flatten(bias_weight))

		if activation_function == "sigmoid":
			result = self.sigmoidNeuronActivation(biased_weighted_sum)
		elif activation_function == "relu":
			result = self.reluNeuronActivation(biased_weighted_sum)

		return Array(result)



	def reluNeuronActivation(self, weighted_sum_vector):
		"""
			Activated neurons using the relu activation function
			
			Arguments: 
			input_vector (Vector) 	:	the weighted sum

			returns (Vector)
		"""
		output_vector = []

		for input_value in weighted_sum_vector:
			output_vector.append(self.relu(input_value))

		return output_vector



	def sigmoidNeuronActivation(self, input_vector):
		"""
			Handles neuron activation

			Arguments: 
			input_vector (Matrix) 	: 	Expects the matrix of weighted sum 
			
			Returns (Matrix)
		"""

		result = []
		for input_val in input_vector:
			result.append(self.sigmoidFunction(input_val))

		return result


	def getWeightedSum(self, input_arr, weight_arr):
		"""
			Caculate weighted sum of the incoming input

			Arguments: 
			input_arr (Array) 	: 	Inputs eigther from a layer ouput or the main testing data
			weight_arr (Array)	: 	The generated weight
			
			Returns (Array) : Weighted sum 
		"""

		weighted_sum_arr = []
		for row in weight_arr:
			sum_of_product = 0
			for index in range(len(row)):
				sum_of_product += (row[index] * input_arr[index])

			weighted_sum_arr.append(sum_of_product)

		return weighted_sum_arr


	def applyBias(self, bias_weight_arr, weighted_sum_arr):
		"""
			apply the bias to the incoming inputs to the recieving neurons layer

			Arguments: 
			bias_weight_arr (Array) 		: 	weights of the bias to be added to the incoming inputs
			weighted_sum_arr (Array)		: 	The generated weight
			
			Returns (Array) : biased inputs
		"""
		return Array(weighted_sum_arr).add(bias_weight_arr)















class WeightUpdates():
	def __init__(self):
		super().__init__()


	def sigmoidWeightCalculation(self, alpha, fwd_l_delta, prev_l_output, init_weight):
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
			weight_ajustments = -learning_rate * [vectorDotProduct(fwd_l_delta, prev_l_output)]
		"""
		neighbor_neuron_dprod = self.vectorDotProduct(fwd_l_delta, prev_l_output)

		weight_adjustment_matrix = []
		for selected_row in neighbor_neuron_dprod:
			result_row = []

			for col_val in selected_row:
				product = alpha * col_val
				result_row.append(product)

			weight_adjustment_matrix.append(result_row)


		weight_update_matrix = self.applyWeightAdjustment(
						initial_weight = init_weight, 
						weight_adjustment = weight_adjustment_matrix, 
						operation = "+"
					)

		return weight_update_matrix



	def sigmoidL2RegularizationWeightUpdate(self, learning_rate, delta_n, prev_layer_output, l2_lambda, intial_weight):
		"""
			Apply the l2 regularization when calculating the weights of the layers

			Aguemtns:
			learning_rate (float)			: The models learning rate
			delta_n (Vector)				: The strenght of the hidden layer recieving from the weihts being update
			prev_layer_output (Matrix)		: The output of the sactivation function of the previos layer
			l2_lambda (float) 				: L2 penalty
			intial_weight (matrix)			: The initial weight

			return matrix with values representing the amount to change the weight
 
		"""

		# delta_w = alpha * (delta * X.T) + lambda * W
		delta_w = self.matrixAddition(
								Array(delta_n).vectorMultiply(Array(prev_layer_output).multiply(learning_rate)), 
								self.matrixScalarMultiply(intial_weight, l2_lambda)
								)
		

		# W = W - (alpha * dW + lambda * W)
		weight_matrix_update = self.matrixSubtract(
								intial_weight,
								self.matrixAddition(
									self.matrixScalarMultiply(delta_w, learning_rate), 
									self.matrixScalarMultiply(intial_weight, l2_lambda)
									)
								) 

		return Array(weight_matrix_update)





	def reluWeightCalculations(self, learning_rate, fwd_l_delta, prev_l_output, init_weight):
		"""
			calculate and update the weights without any regularization
			
			Equation: 	dW = learning_rate * delta[l] [x] X.T)
						W = W - (alpha * dw)
		"""
		weight_adjustment = self.matrixScalarMultiply(self.vectorDotProduct(fwd_l_delta, prev_l_output), learning_rate)
		weight_update = self.matrixSubtract(init_weight, self.matrixScalarMultiply(weight_adjustment, learning_rate)) 

		return weight_update




	def applyWeightAdjustment(self, initial_weight, weight_adjustment, operation = "+"):
		"""
			Apply the adjustments of the weights to the initial weight to update its value by getting the sum of the two array

			Arguments:
			initial_weight (List / Array)			:	The weights value that is used in forward propagation
			weight_adjustment  (List / Array)		:	The value used to add to the initial weight

			Returns: Array
		"""
		if operation == "+":
			returned_value = self.matrixAddition(initial_weight, weight_adjustment)
		elif operation == "-":
			returned_value = self.matrixSubtract(initial_weight, weight_adjustment)

		return Array(returned_value)










class DeltaCalculationMethods():
	def __init__(self):
		super().__init__()


	# rename 
	# preceding_neuron_output_vector to l_output
	# weight_matrix 
	# proceding_neuron_output_matrix to fwd_l_delta

	def sigmoidDeltaCalculation(self, l_output, weight_matrix, fwd_l_delta):
		"""
			Calculate the delta of the a layer using sigmoid derivative

			Arguments: 
				l_output (vector) 				:	The output of the layer or the ouput of the sigmoid function in that layer
				weight_matrix (matrix)			:	Updated weight matrix next to the current layer being update
				proceding_neuron_output_matrix (matrix)  (ith_layer + 1) 

			Return (Vertor) Returns the calculated delta of the layer

			Equation : transpose(weight_matrix) matx_mult(fwd_l_delta) * (l_output (1 - l_output)))
		"""


		"""
		neuron_strenghts = self.vectorMultiply(
			self.flatten(self.getMatrixSumOfRow(self.matrixVectorMultiply(self.transpose(weight_matrix), fwd_l_delta))), 
			ArrayMethods().vectorMultiply(l_output, Array(l_output).subtract(1))
			)
		"""

		transposed_weight = self.transpose(weight_matrix)

		subtracted_arr = []
		for neuron_val in l_output:
			subtracted_arr.append(1 - neuron_val)

		product_arr = []
		for index in range(len(l_output)):
			product_arr.append(l_output[index] * subtracted_arr[index])

		dot_product_arr = self.matrixVectorMultiply(transposed_weight, fwd_l_delta)
		sum_of_rows_arr = self.getMatrixSumOfRow(dot_product_arr)
		neuron_strenghts = self.vectorMultiply(self.flatten(sum_of_rows_arr), product_arr)


		return Array(neuron_strenghts)



	def reluDeltaCalculation(self, l_fl_weight_matrix, fwd_l_delta, l_output):
		"""
			Calculate the delta of a layer using the dewrivative of the relu activation function

			Arguments:
			l_fl_weight_matrix (matrix) 		:	The weight matrix in the middle of the current layer and the next layer
			fwd_l_delta (vecctor)				:	The calculated delta of the layer in front of the current layer
			l_output (Vector)					: 	The ouput of the current layer

			Return: (Vector)
			Equation : dot(l_fl_weight_matrix, fwd_l_delta) * (l_output > 0)
		"""

		dy_l_output = []

		for value in l_output:
			if value > 0:
				dy_l_output.append(1)
			elif value <= 0:
				dy_l_output.append(0)


		delta_vector = ArrayMethods().vectorMultiply(
			ArrayMethods().matrixVectorDotProd(
				Array(l_fl_weight_matrix).transpose(), 
				fwd_l_delta), 
				dy_l_output
			)

		return delta_vector






class BackPropagation(ArrayMethods, WeightUpdates, DeltaCalculationMethods):
	def __init__(self, learning_rate = -0.01):
		"""
			This class handles the backpropagation acalculation methods
		"""
		super().__init__()
		self.learning_rate = learning_rate


	def getCrossEntropyLoss(self, predicted_ouputs_vector, actual_label_vector):
		"""
			This method is made to calculate the coss entropy loss for the final layer

			Arguments:
			predicted_ouputs_vector (Vector) (p)		:	Networks final layer or prediction
			actual_label_vector (Vector) (y)	:	The actual label for the given problem

			Return (Vector) calculated loss
			Equation : -y * log(p) - (1 - y) * log(1-p)
		"""
		try:
			output_vector = []

			for value_index in range(len(predicted_ouputs_vector)):
				y = actual_label_vector[value_index]
				p = predicted_ouputs_vector[value_index]

				output = -y * log(p+1e-9) - (1 - y) * log(1 - p+1e-9)

				output_vector.append(output)

			return output_vector
		except ValueError:
			err_msg = "Math dommain erro: where p = " + str(p)
			raise Exception(err_msg)



	def getFinalLayerDelta(self, predicted_ouputs_vector, actual_label_vector):
		"""
			Calculate the final layer neuron strenghts

			Arguments:
			predicted_ouputs_vector (List / Array)				:	Final output that is calculated by sigmoid function
			actual_label_vector (List / Array)	:	The final ouput that is produced by argmax function

			Returns: Array

		"""
		#returned_value = Array(self.vectorSubtract(predicted_ouputs_vector, actual_label_vector)).squared()
		returned_value = self.vectorSubtract(predicted_ouputs_vector, actual_label_vector)
		return Array(returned_value)





	def updateLayerWeight(self, alpha, fwd_l_delta, prev_l_output, init_weight, activation_function_method):
		
		"""
			Update weight matrix

		"""
		
		if activation_function_method == "sigmoid":
			if len(fwd_l_delta) != 0 and len(prev_l_output) != 0 and len(init_weight) != 0:
				weight_update_matrix = self.sigmoidWeightCalculation(
											alpha = alpha, 
											fwd_l_delta = fwd_l_delta, 
											prev_l_output = prev_l_output, 
											init_weight = init_weight
											)

		elif activation_function_method == "relu":
			if alpha != 0 and len(fwd_l_delta) != 0 and len(prev_l_output) != 0 and len(init_weight) != 0:
				weight_update_matrix = self.reluWeightCalculations(
											learning_rate = alpha, 
											fwd_l_delta = fwd_l_delta, 
											prev_l_output = prev_l_output, 
											init_weight = init_weight
											)

		return Array(weight_update_matrix)
		if len(weight_update_matrix) != 0: 
			return Array(weight_update_matrix)
		else:
			raise ValueError("Error in mthod updateLayerWeight: returning none")




	def L2regularizedWeightUpdate(self, learning_rate, delta_n, prev_layer_output, l2_lambda, intial_weight, activation_function = "sigmoid"):
		"""
			Update weight matrix with a regularization method

		"""
		if activation_function == "sigmoid":
			weight_update_matrix = self.sigmoidL2RegularizationWeightUpdate(learning_rate, delta_n, prev_layer_output, l2_lambda, intial_weight)

		return weight_update_matrix



	# rename arguments;
	# preceding_neuron_output_arr to l_output
	# proceding_neuron_output_arr to fwd_l_delta
	def getHiddenLayerDelta(self, l_output, weight, fwd_l_delta, activation_function = "sigmoid"):
		"""
			calculate the strenght of the neurons in a hidden layer 
			
			Arguments:
			prev_l_output (List / Array)				:	The layer of neurons that is first to recieve data relative to forward propagation direction
			weights (List / Array) 						:	The weights in the middle to the two given neurons
			proceding_neuron_strenght (List / Array)	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			
			Retuns: Vector

		"""
		if activation_function == "sigmoid":
			delta_vector = self.sigmoidDeltaCalculation(
								l_output = l_output, 
								weight_matrix = weight, 
								fwd_l_delta = fwd_l_delta
								)

		elif activation_function == "relu":
			delta_vector = self.reluDeltaCalculation(
								l_fl_weight_matrix = weight, 
								fwd_l_delta = fwd_l_delta, 
								l_output = l_output
							)
		

		return delta_vector


	def adjustBiasWeight(self, neuron_strnght):
		"""
			Calculate bias adjustment
			
			Argumemts:
			neuron_strnght	(List / Array)	:	Updated neuron strenghts

			Formula: -learning_rate * updated_neuron_strenght
			
			Return Array
		"""
		adjusted_biase = Array(neuron_strnght).multiply(self.learning_rate)
		return Array(adjusted_biase)
		



	def getMeanSquaredError(self, ouput, labeld_output):
		"""
			Calculate the mean squared error or cost value

			Arguments;
			ouput (List / Array) 				:	The unlabled output, or the output from the sigmoid function
			labeld_output (List / Array)		:	The labled output
			
			returns : float
			Formula : 1 / lne(ouput) * sum((ouput - labeld_output) ** 2)

		"""

		arr_difference = self.vectorSubtract(ouput, labeld_output)
		squared_arr = self.vectorSquare(arr_difference)

		arr_sum = Array(squared_arr).sum()
		e = 1 / 3 * arr_sum

		return e










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
			for training_batch_set_index in range(batches_count):
				# The set of trainig data in the batch array
				training_batch = self.batch_array[training_batch_set_index]
				# The answer key or labeled ouput of the batch
				batch_key = self.answer_key_batch_array[training_batch_set_index]

				# loop through the entire data inside the batch 
				count_data_in_batch = len(training_batch)
				for data_index in range(count_data_in_batch):
					# get the input data for the current loop
					input_data = training_batch[data_index]
					# get the labeld output of the input data
					input_labeld_data = batch_key[data_index]



					#### FORWARD PROPAGATION ####
					# holds the value of the ouputs of the previous layer the training_data variable is added intialy as it would 
					# represent as the final ouput for the first layer in back propagation
					layer_ouputs_matrix = []
					
					# The input of the current layer
					current_layer_input = input_data

					# Loop through the entire layer of the neural network
					for layer_index in range(layer_count):
						layer_activation_function = self.layer_size_vectors[layer_index][1]
						# create a layer where neuron activation and other transformation will handle
						layer_ouput = self.fowardPropagation(
										input_value = current_layer_input, 
										weight_value = self.weights_set[layer_index], 
										bias_weight = self.bias_weight_set[layer_index],
										activation_function = layer_activation_function
										)

						# Append the output of the layer for backpropagation
						layer_ouputs_matrix.append(layer_ouput)
						# update the input for the next layer
						current_layer_input = layer_ouput


					#### calculate Loss functions ####
					mean_square_error = self.getMeanSquaredError(
						 						ouput = layer_ouputs_matrix[-1], 
						 						labeld_output = input_labeld_data
						 						)
					# Append the result to the error log
					self.mean_square_error_log.append(mean_square_error)
					# Check if the model prediction was correct
					final_prediction = ActivationFunction().argMax(layer_ouputs_matrix[-1])
					if final_prediction == input_labeld_data:
						correct_prediction += 1




					#### BACK PROPAGATION #####
					# delta of the current layer in loop, Initial value was calculated using the final layer strenght
					delta_h = self.getFinalLayerDelta(
											predicted_ouputs_vector = layer_ouputs_matrix[-1], 
											actual_label_vector = input_labeld_data
										)

					# Loop throught the entire layer from start to the final
					for layer_index in range(len(self.layer_size_vectors) - 1, -1, -1): 
						# get the specific activation function use by the current layer in iteration
						layer_activation_function = self.layer_size_vectors[layer_index][1]
						# update the weight of the biases every layer using the delta_h
						self.bias_weight_set[layer_index] = self.adjustBiasWeight(neuron_strnght = delta_h)

						# check if the layer in proccess is a hidden layer
						if layer_index != 0:
							# if the regularization_method is set to L2 regularization method
							if self.regularization_method == "L2":
								# calculate the adjustment needed to the weight with the L2 regualrization equation
								weight_update = self.L2regularizedWeightUpdate(
																learning_rate = self.learning_rate, 
																delta_n = delta_h, 
																prev_layer_output = layer_ouputs_matrix[layer_index - 1], 
																l2_lambda = self.l2_penalty, 
																intial_weight = self.weights_set[layer_index],
																activation_function = layer_activation_function
															)

							# if there is no regularization method used the the weight is calculated using the sigmoid derivative in calculateWeightAdjustment method
							elif self.regularization_method == "none":
								# calculate the adjustment needed to apply to the initial weight
								
								weight_update = self.updateLayerWeight(
														alpha = self.learning_rate,
														fwd_l_delta = delta_h, 
														prev_l_output = layer_ouputs_matrix[layer_index - 1], 
														init_weight = self.weights_set[layer_index],
														activation_function_method = layer_activation_function
													)


							# Update the layer weight with the new calculated weight
							self.weights_set[layer_index] = weight_update
							layer_strenght = self.getHiddenLayerDelta(
										l_output = layer_ouputs_matrix[layer_index - 1], 
										weight = weight_update, 
										fwd_l_delta = delta_h,
										activation_function = layer_activation_function
									)

							# update the value of delta_h coming from the calculated value of the hidden layer strenghts
							delta_h = layer_strenght


						## check if the current layer in interation is the main input layer ##
						elif layer_index == 0:
							if self.regularization_method == "L2":
								weight_update = self.L2regularizedWeightUpdate(
																learning_rate = self.learning_rate, 
																delta_n = delta_h, 
																prev_layer_output = input_data, 
																l2_lambda = self.l2_penalty, 
																intial_weight = self.weights_set[layer_index],
																activation_function = layer_activation_function
															)
								
							elif self.regularization_method == "none":
								weight_update = self.updateLayerWeight(
														alpha = self.learning_rate,
														fwd_l_delta = delta_h, 
														prev_l_output = input_data, 
														init_weight = self.weights_set[layer_index],
														activation_function_method = layer_activation_function
													)

							self.weights_set[layer_index] = weight_update
							break


		# After fitting calculate the model acccuracy
		self.calculateModelAccuracy(
				n_of_correct_pred = correct_prediction, 
				n_of_training_data = len(training_data), 
				training_epoch = epoch
			)

		self.printFittingSummary()



	def predict(self, input_data):
		layer_input = input_data
		layer_output_arr = []

		for layer_index in range(len(self.layer_sizes)):
			# create a layer where neuron activation and other transformation will handle
			layer_ouput = self.fowardPropagation(
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