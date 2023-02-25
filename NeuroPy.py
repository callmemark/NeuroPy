
"""

Author: Mark John Velmonte
Date: February 2, 2023

Description: Contains class to create simple but expandable neural network from scratch.

"""

from random import uniform





class Math():
	def __init__(self):
		pass

	def getSqrt(self, num, decimal_place = 4):
	    """
	    Calculates the square root of a given number

	    Parameters:
	    num (float)			:	The number whose square root needs to be calculated.
	    decimal_place(int)	:	The amount of decimal places before roundoff

	    Returns:
	    float: The square root of the given number.

	    """
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
		standard_dev = Math().getSqrt(self.subtract(self.mean()).squared().sum() / len(self))
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








class WeightInitializationMethods(Math):
	def __init__(self):
		"""
			Methods to intializ random value generated using different mathematical functions

			Arguments: Takes 0 arguments
		"""
		super().__init__()



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

		lower_range, upper_range = -(self.getSqrt(6.0) / self.getSqrt(sum_of_node_count)), (self.getSqrt(6.0) / self.getSqrt(sum_of_node_count))
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
		result = 1 / (1 + self.E ** -x)
		return result


	def argMax(selc, arr):
		"""
			This method search for the maximum value and create a new list where only the maximum value will have a value of 1

			Arguments: 
			arr(Array) 	: The array that will be transformed into a new array
			
			Returns: Array
		"""
		output_array = []

		max_value_index = arr.index(max(arr))

		for index in range(len(arr)):
			if index == max_value_index:
				output_array.append(1)
			elif index != max_value_index:
				output_array.append(0)

		return Array(output_array)










class ForwardPropagation(ActivationFunction):
	def __init__(self):
		"""
			This class contains different methods for neural network forward propagation

			Arguments: takes 0 arguments
		"""
		super().__init__()


	def createLayer(self, input_value, weight_value, bias_weight):
		"""
			Creates a nueral network layer

			Arguments: 
			input_value (Array) 	: 	testing inputs 
			weight_value (Array)	:	The corresponding weight to this layer
			bias_weight (Array)		:	The weight of the bias for this layer
			
			Returns (Array) : The ouput of the this layer
		"""
		weighted_sum = self.getWeightedSum(input_value, weight_value)
		biased_weighted_sum = Array(weighted_sum).addArray(arrayMethods().flatten(bias_weight))
		result = self.neuronActivation(biased_weighted_sum)

		return Array(result)


	def neuronActivation(self, input_array):
		"""
			Handles neuron activation

			Arguments: 
			input_array (Array) 	: 	Expects the array of weighted sum 
			
			Returns (Array)
		"""
		result = []
		for input_val in input_array:
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









class arrayMethods():
	def __init__(self):
		pass


	def dot_prod(self, multiplier_arr, arr_to_mult):
		output_array = []

		for multiplier in multiplier_arr:
			row_arr = []
			for value in arr_to_mult:
				product = multiplier * value
				row_arr.append(product)

			output_array.append(row_arr)

		return output_array


	def multidim_arr_dotprod(self, multiplicand_arr, multiplier_arr):
		output_array = []

		for selected_row in multiplicand_arr:
			new_row = []
			for index in range(len(selected_row)):
				new_row.append(selected_row[index] * multiplier_arr[index])
			output_array.append(new_row)


		return output_array



	def getSumOfRows(self, _2d_array):
		# get and return the sum of each rows of a 2d array 
		final_arr = []

		for selected_row in _2d_array:
			sum_of_current_iter = 0
			for value in selected_row:
				sum_of_current_iter += value

			final_arr.append([sum_of_current_iter])

		return final_arr



	def subtractArray(self, arr_01, array_02):
		## subtact values of a 1 dimensional array
		arr_01_lenght = len(arr_01)
		arr_02_lenght = len(arr_02)

		if arr_01_lenght != arr_02_lenght:
			raise Exception(str("Error: Array sized are not equal where " + str(arr_01_lenght) + " != " + str(arr_02_lenght)))

		difference_array = []
		loop_count = int((array_01_lenght + arr_02_lenght) / 2)

		for index in range(loop_count):
			difference_array.append(array_01[index] - arr_02[index])

		return difference_array



	def multiply_array(self, array_01, array_02):
		array_01_lenght = len(array_01)
		array_02_lenght = len(array_02)

		if array_01_lenght != array_02_lenght:
			raise Exception("Error: Array are not equal where size is ", array_01_lenght, " != ", array_02_lenght)

		output_array = []
		loop_n = int((array_01_lenght + array_02_lenght) / 2)

		for index in range(loop_n):
			output_array.append(array_01[index] * array_02[index])

		return output_array



	def addArray(self, arr1, arr2):
	    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
	        raise ValueError("Arrays must have the same shape")

	    result = [[0 for _ in range(len(arr1[0]))] for _ in range(len(arr1))]

	    for i in range(len(arr1)):
	        for j in range(len(arr1[0])):
	            result[i][j] = arr1[i][j] + arr2[i][j]

	    return result


	def subtractArray(self, arr1, arr2):
		if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
			raise ValueError("Arrays must have the same shape")

		result = [[0 for _ in range(len(arr1[0]))] for _ in range(len(arr1))]

		for i in range(len(arr1)):
			for j in range(len(arr1[0])):
				result[i][j] = arr1[i][j] - arr2[i][j]

		return result


	def subtractOneDimArray(self, minuend_array, subtrahend_array):
		# Subtract a flatten array
		subtracted_arr = []
		minuend_arr_size = len(minuend_array)
		subtrahend_arr_size = len(subtrahend_array)

		if minuend_arr_size != subtrahend_arr_size:
			raise Exception(str("Error on function 'subtractOneDimArray'. Arrays are not equal lenght with sizes " + str(minuend_arr_size) + " and " + str(subtrahend_arr_size)))

		index_count = int((minuend_arr_size + subtrahend_arr_size) / 2)
		# Loop through the lists
		for index in range(index_count):
			subtracted_arr.append(minuend_array[index] - subtrahend_array[index])

		# returns an array with the calculated difference
		return subtracted_arr



	def flatten(self, arr):
	    flattened = []

	    if len(self.getShape(arr)) == 1:
	    	return arr

	    for element in arr:
	        if isinstance(element, list):
	            flattened.extend(self.flatten(element))
	        else:
	            flattened.append(element)

	    return flattened


	def getShape(self, array_arg):
	    shape = []

	    while isinstance(array_arg, list):
	        shape.append(len(array_arg))
	        array_arg = array_arg[0]

	    return shape


	def transpose(self, arr):
	    shape = self.getShape(arr)
	    transposed_list = [[None]*shape[0] for _ in range(shape[1])]
	    
	    for i in range(shape[0]):
	        for j in range(shape[1]):
	            transposed_list[j][i] = arr[i][j]
	    
	    return transposed_list


	def square(self, arr):
		squared_arr = []
		arr_shape = self.getShape(arr)
		if len(arr_shape) != 1:
			raise Exception("Error in function square, Array should be one dimesion but have " + str(arr_shape))

		for value in arr:
			squared_arr.append(value ** 2)

		return squared_arr



	def weightMultiplyArr(self, arr, int_multiplier):
		resulting_arr = []

		for value in arr:
			resulting_arr.append(value * int_multiplier)

		return resulting_arr








class Backpropagation(arrayMethods, Array):
	def __init__(self, learning_rate = -0.01):
		"""
			This class handles the backpropagation acalculation methods
		"""

		super().__init__()
		self.learning_rate = learning_rate


	def getFLayerNeuronStrenght(self, final_output, argmaxed_final_output):
		"""
			Calculate the final layer neuron strenghts

			Arguments:
			final_output (List / Array)				:	Final output that is calculated by sigmoid function
			argmaxed_final_output (List / Array)	:	The final ouput that is produced by argmax function

			Returns: Array

		"""
		returned_value = self.subtractOneDimArray(final_output, argmaxed_final_output)
		return Array(returned_value)


	def applyWeightAdjustment(self, initial_weight, weight_adjustment):
		"""
			Apply the adjustments of the weights to the initial weight to update its value by getting the sum of the two array

			Arguments:
			initial_weight (List / Array)			:	The weights value that is used in forward propagation
			weight_adjustment  (List / Array)		:	The value used to add to the initial weight

			Returns: Array
		"""
		returned_value = self.addArray(initial_weight, weight_adjustment)
		return Array(returned_value)


	def calculateWeightAdjustment(self, proceding_neuron_strenght, preceding_neuron_output):
		""" 
			Calculate and return an array of floats that is intended to use for calibrating the weights of the 
			Neural network
			
			Arguments:
			proceding_neuron_strenght (List / Array)	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			preceding_neuron_output (List / Array)		:	The layer of neurons that is first to recieve data relative to forward propagation direction
			
			Returns: Array

			formula:
			weight_ajustments = -learning_rate * [dot_prod(proceding_neuron_strenght, preceding_neuron_output)]

		"""

		neighbor_neuron_dprod = self.dot_prod(proceding_neuron_strenght, preceding_neuron_output)

		final_weight = []
		for selected_row in neighbor_neuron_dprod:
			result_row = []

			for col_val in selected_row:
				product = self.learning_rate * col_val
				result_row.append(product)

			final_weight.append(result_row)

		return Array(final_weight)


	def getHLayerNeuronStrength(self, preceding_neuron_output_arr, weight, proceding_neuron_output_arr):
		"""
			calculate the strenght of the neurons in a hidden layer 
			
			Arguments:
			preceding_neuron_output (List / Array)		:	The layer of neurons that is first to recieve data relative to forward propagation direction
			weights (List / Array) 						:	The weights in the middle to the two given neurons
			proceding_neuron_strenght (List / Array)	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			
			Retuns: Array

		"""

		transposed_weight = self.transpose(weight)

		subtracted_arr = []
		for neuron_val in preceding_neuron_output_arr:
			subtracted_arr.append(1 - neuron_val)

		product_arr = []
		for index in range(len(preceding_neuron_output_arr)):
			product_arr.append(preceding_neuron_output_arr[index] * subtracted_arr[index])

		dot_product_arr = self.multidim_arr_dotprod(transposed_weight, proceding_neuron_output_arr)
		sum_of_rows_arr = self.getSumOfRows(dot_product_arr)
		neuron_strenghts = self.multiply_array(self.flatten(sum_of_rows_arr), product_arr)

		return Array(neuron_strenghts)


	def getAdjustedBiasdWeights(self, corresponding_layer_neurons):
		"""
			Calculate bias adjustment
			
			Argumemts:
			corresponding_layer_neurons	(List / Array)	:	Updated neuron strenghts

			Formula: -learning_rate * updated_neuron_strenght
			
			Return Array
		"""
		#print(Array(corresponding_layer_neurons).multiply(self.learning_rate))
		adjusted_biase = Array(corresponding_layer_neurons).multiply(self.learning_rate)
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

		arr_difference = self.subtractOneDimArray(ouput, labeld_output)
		squared_arr = self.square(arr_difference)

		arr_sum = Array(squared_arr).sum()
		e = 1 / 3 * arr_sum

		return e