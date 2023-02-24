
"""

Author: Mark John Velmonte
Date: February 2, 2023

Description: Contains class to create simple but expandable neural network from scratch.

"""

from random import uniform





def Math():
	def __init__(self):
		pass







class WeightInitializationMethods():
	def __init__(self):
		pass

	def radomInitializer(self, min_f = 0, max_f = 1.0):
		rwg = 2 * uniform(min_f, max_f) - 1

		return rwg





class WeightInitializer(WeightInitializationMethods):
	def __init__(self):
		super().__init__()


	def intializeWeight(self, dim, min_f = -1.0, max_f = 1.0):
		final_weight_arr = []
		row = dim[0]
		col = dim[1]

		for i in range(row):
			col_arr = []
			for j in range(col):
				col_arr.append(self.radomInitializer(min_f, max_f))

			final_weight_arr.append(col_arr)

		return final_weight_arr






class ActivationFunction():
	def __init__(self):
		self.E = 2.71


	def sigmoidFunction(self, x):
		result = 1 / (1 + self.E ** -x)
		return result


	def argMAx(selc, arr):
		output_array = []

		max_value_index = arr.index(max(arr))

		for index in range(len(arr)):
			if index == max_value_index:
				output_array.append(1)
			elif index != max_value_index:
				output_array.append(0)

		return output_array










class ForwardPropagation(ActivationFunction):
	def __init__(self):
		super().__init__()


	def createLayer(self, input_value, weight_value):
		weighted_sum = self.getWeightedSum(input_value, weight_value)
		result = self.neuronActivation(weighted_sum)

		return result


	def neuronActivation(self, input_array):
		result = []
		for input_val in input_array:
			result.append(self.sigmoidFunction(input_val))

		return result


	def getWeightedSum(self, input_arr, weight_arr):
		weighted_sum_arr = []
		for row in weight_arr:
			sum_of_product = 0
			for index in range(len(row)):
				sum_of_product += (row[index] * input_arr[index])

			weighted_sum_arr.append(sum_of_product)

		return weighted_sum_arr








class GeneticMutation():
	def __init__(self):
		super().__init__()


	def geneticMutationAlgorithm(self, weight_set_A, weight_set_B, index_lim_arr = []):
		weight_seta_size = len(weight_set_A)
		weight_setb_size = len(weight_set_B)

		weight_nrow = int((weight_seta_size + weight_setb_size) / 2)

		mutated_weight_set = []
		
		for row_index in range(weight_nrow):
			set_a_col = weight_set_A[row_index]
			set_b_col = weight_set_B[row_index]
			new_mutated_col = []


			if len(index_lim_arr) > 0:
				if row_index in index_lim_arr:
					continue


			col_size = int((len(set_a_col) + len(set_b_col)) / 2)

			for col_index in range(col_size) :
				rng = uniform(0, 1)
				if rng >= 0.5:
					new_mutated_col.append(set_a_col[col_index])
				elif rng < 0.5:
					new_mutated_col.append(set_b_col[col_index])

			mutated_weight_set.append(new_mutated_col)

		return mutated_weight_set









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







class Array(list):
	def __init__(self, data):
		super().__init__(data)
		#self.shape = (len(data),)
		self.shape = self.getShape()


	def transpose(self):
		arr = self
		shape = self.getShape()
		transposed_list = [[None]*shape[0] for _ in range(shape[1])]

		for i in range(shape[0]):
			for j in range(shape[1]):
				transposed_list[j][i] = arr[i][j]

		return Array(transposed_list)


	def getShape(self):
	    shape = []
	    arr = self

	    while isinstance(arr, list):
	        shape.append(len(arr))
	        arr = arr[0]

	    self.shape = shape
	    return shape


	def multiply(self, multiplier):
		array_product = []

		for value in self:
			array_product.append(value * multiplier)

		return Array(array_product)


	def sumOfOneAxis(self):
		result = 0

		for value in self:
			result += value

		return result








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
			final_output 				:	Final output that is calculated by sigmoid function
			argmaxed_final_output		:	The final ouput that is produced by argmax function

			Returns: Array

		"""

		return self.subtractOneDimArray(final_output, argmaxed_final_output)


	def applyWeightAdjustment(self, initial_weight, weight_adjustment):
		"""
			Apply the adjustments of the weights to the initial weight to update its value by getting the sum of the two array

			Arguments:
			initial_weight				:	The weights value that is used in forward propagation
			weight_adjustment 			:	The value used to add to the initial weight

			Returns: Array
		"""
		return self.addArray(initial_weight, weight_adjustment)


	def calculateWeightAdjustment(self, proceding_neuron_strenght, preceding_neuron_output):
		""" 
			Calculate and return an array of floats that is intended to use for calibrating the weights of the 
			Neural network
			
			Arguments:
			proceding_neuron_strenght	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			preceding_neuron_output 	:	The layer of neurons that is first to recieve data relative to forward propagation direction
			
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

		return final_weight


	def getHLayerNeuronStrength(self, preceding_neuron_output_arr, weight, proceding_neuron_output_arr):
		"""
			calculate the strenght of the neurons in a hidden layer 
			
			Arguments:
			preceding_neuron_output 	:	The layer of neurons that is first to recieve data relative to forward propagation direction
			weights 					:	The weights in the middle to the two given neurons
			proceding_neuron_strenght	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			
			Retuns:Array

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

		return neuron_strenghts


	def getAdjustedBiased(self, corresponding_layer_neurons):
		"""
			Calculate bias adjustment
			
			Argumemts:
			corresponding_layer_neurons	:	Updated neuron strnghts

			Formula: -learning_rate * updated_neuron_strenght
			
		"""

		adjusted_biase = self.weightMultiplyArr(corresponding_layer_neurons, self.learning_rate )
		return adjusted_biase
		


	def getMeanSquaredError(self, ouput, labeld_output):
		"""
			Calculate the mean squared error or cost value

			Arguments;
			ouput 					:	The unlabled output, or the output from the sigmoid function
			labeld_output			:	The labled output from the sigmoid funtion

			returns : float
			Formula : 1 / lne(ouput) * sum((ouput - labeld_output) ** 2)

		"""

		arr_difference = self.subtractOneDimArray(ouput, labeld_output)
		squared_arr = self.square(arr_difference)

		arr_sum = Array(squared_arr).sumOfOneAxis()
		e = 1 / 3 * arr_sum

		return e





    






sample_input = [0.1, 0.89, 0.87, 0.9, 0.2]


l1_w = WeightInitializer().intializeWeight([20, 5])
l2_w = WeightInitializer().intializeWeight([15, 20])
fl_w = WeightInitializer().intializeWeight([3, 15])

l1_output = ForwardPropagation().createLayer(sample_input, l1_w)
l2_output = ForwardPropagation().createLayer(l1_output, l2_w)
fl_output = ForwardPropagation().createLayer(l2_output, fl_w)


argmax_putput = ActivationFunction().argMAx(fl_output)

Backpropagation = Backpropagation(learning_rate = -0.01)

final_layer_neuron_cost = Backpropagation.getFLayerNeuronStrenght(fl_output, argmax_putput)


weight_adjustments = Backpropagation.calculateWeightAdjustment(final_layer_neuron_cost, l2_output)
applied_weight_adjustment = Backpropagation.applyWeightAdjustment(fl_w, weight_adjustments)
middle_layer_neuron_cost = Backpropagation.getHLayerNeuronStrength(l2_output, applied_weight_adjustment, final_layer_neuron_cost)


weight_adjustments_2 = Backpropagation.calculateWeightAdjustment(middle_layer_neuron_cost, l1_output)
applied_weight_adjustment_2 = Backpropagation.applyWeightAdjustment(l2_w, weight_adjustments_2)
middle_layer_neuron_cost_2 = Backpropagation.getHLayerNeuronStrength(l1_output, applied_weight_adjustment_2, middle_layer_neuron_cost)


weight_adjustments_3 = Backpropagation.calculateWeightAdjustment(middle_layer_neuron_cost_2, sample_input)
applied_weight_adjustment_3 = Backpropagation.applyWeightAdjustment(l1_w, weight_adjustments_3)
middle_layer_neuron_cost_3 = Backpropagation.getHLayerNeuronStrength(sample_input, applied_weight_adjustment_3, middle_layer_neuron_cost_2)

error = Backpropagation.getMeanSquaredError(fl_output, argmax_putput)

print(error)

