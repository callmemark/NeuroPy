"""
Author: Mark John Velmonte

Description:
	Extend pythons list functionality and add vector operations
	This module is made to work with NeurPy module but can work with other codes

"""


class Vector(list):
	def __init__(self, data):
		"""
			Create new Vector object to extend Python list functionality
		"""
		super().__init__(data)


	# from multiply to multiplyScalar
	def multiplyScalar(self, multiplier):
		"""
			Multiply this (Vector) object

			Arguments:
			multiplier(scalar)		:	The number use to multiply each element in this Vector

			Return: Vector
		"""
		vector_product = []
		for value in self:
			vector_product.append(value * multiplier)

		return Vector(vector_product)


	#from add to addScalar()
	def addScalar(self, addends):
		"""
			Add this (Vector) object

			Arguments:
			addends(scalar)

			Return: Vector
		"""
		vector_arr = []
		for value in self:
			vector_arr.append(value + addends)

		return Vector(vector_arr)


	# from subtract to subtractScalar
	def subtractScalar(self, subtrahend):
		"""
			Subtract this (Vector) object

			Arguments:
			subtrahend(scalar)

			Return: Vector

		"""
		vector_arr = []
		for value in self:
			vector_arr.append(value - subtrahend)

		return Vector(vector_arr)


	def devideScalar(self, devider):
		"""
			Subtract this (Vector) object

			Arguments:
			subtrahend(scalar)

			Return: Vector
		"""
		vector_arr = []
		for value in self:
			vector_arr.append(value / devider)

		return Vector(vector_arr)


	def sum(self):
		"""
			get the sum of all alements in vector

			Arguments: takes 0 arguments
			Return: float
		"""
		total = 0
		for value in self:
			total += value

		return total


	def min(self):
		"""
			get the minimum or lowest value in this (Vector) object

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
			get the maximum or highest value in this (Vector) object

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
			caculkate the mean value of this (Vector) object

			Arguments: takes 0 arguments
			Return: float
		"""
		sum_of_arr = self.sum()
		mean_val = sum_of_arr / len(self)

		return mean_val


	def squared(self):
		"""
			caculkate the squared value of every elements in this (Vector) object

			Arguments: takes 0 arguments
			Return: Vector
		"""
		squared_arr = []
		for value in self:
			squared_arr.append(value ** 2)

		return Vector(squared_arr) 


	def std(self):
		"""
			caculate the standard deviation value of this (Vector) object

			Arguments: takes 0 arguments
			Return: float
		"""
		#standard_dev = Math().sqrt(self.subtract(self.mean()).squared().sum() / len(self))
		standard_dev = sqrt(self.subtract(self.mean()).squared().sum() / len(self))
		return standard_dev


	def maximum(self, sacalar_difference):
		"""
			returns the vector of the elements that have higher value than the sacalar_difference argument of else return the sacalar_difference
		"""	
		output_vector = []

		for val in self:
			output_vector.append(max(sacalar_difference, val))

		return Vector(output_vector)



	def addVector(self, addends_vector):
		"""
			Add vector to this object

			Arguments:
			addends_arr(addends_vector)

			Return: Vector
		"""
		#self.checkLenghtEquality(addends_vector)

		output_vector = []
		for self_val, arg_val in zip(self, addends_vector):
			output_vector.append(self_val + arg_val)

		return Vector(output_vector)


	def subtractVector(self, minuend_vector):
		"""
			subtract vector to this vector

			Arguments:
			minuend_vector

			Return: Vector
		"""
		#self.checkLenghtEquality(minuend_vector)

		output_vector = []
		for self_val, arg_val in zip(self, minuend_vector):
			output_vector.append(self_val - arg_val)

		return Vector(output_vector)



	def multiplyVector(self, multiplier_vector):
		"""
			multiply vector to this vector

			Arguments:
			addends_arr(addends_vector)

			Return: Vector
		"""
		#self.checkLenghtEquality(multiplier_vector)

		output_vector = []
		for self_val, arg_val in zip(self, multiplier_vector):
			output_vector.append(self_val * arg_val)

		return Vector(output_vector)



	def devideVector(self, devidor_vector):
		"""
			devide vector to this vector

			Arguments:
			devidor_vector

			Return: Vector
		"""
		#self.checkLenghtEquality(devidor_vector)

		output_vector = []
		for self_val, arg_val in zip(self, devidor_vector):
			output_vector.append(self_val / arg_val)

		return Vector(output_vector)


	def avgNorm(self):
		"""
		Normlized by getting deviding the value to the sum of the vector
		"""
		vec_sum = sum(self)
		output_vector = []

		for value in self:
			output_vector.append(value / vec_sum)

		return Vector(output_vector)


	def minMaxNorm(self):
		"""
			Normalized the values of the vector

			Return the normlized vector
		"""
		normalized_arr = []

		for value in self:
			norm = (value - self.max()) / (self.max() - self.min())
			normalized_arr.append(norm)

		return Vector(normalized_arr)



	def dotProd(self, vector_argument):
		"""
			Get the dot product of the two vector

			Return Scalar
		"""
		return self.multiplyVector(vector_argument).sum()


	def exp(self):
		"""
			Exponentialized the values fo the vector

			Return Vector
		"""
		E = 2.71828182846 
		output_vector = []

		for value in self:
			output_vector.append(E ** value)

		return Vector(output_vector)


	def average(self):
		"""
			Returns scalar representing the average of the vector
		"""
		return (self.sum() / len(self))


	def checkLenghtEquality(self, given_argument):
		"""
			Check if the given argument gave equal lenght to this Vector
		"""

		if len(given_argument) != len(self):
			err_msg = "The shape of given value " + str(len(given_argument)) + " != to the Vector with size " + str(len(self))
			raise ValueError(err_msg)
		elif len(given_argument) == len(self):
			return True




	def valueWiseMultiplcation(self, multiplier_vector):
		"""
		Return: Matrix

		"""
		
		output_matrix = []

		for value in self:
			row_vector = []
			for element in multiplier_vector:
				row_vector.append(value * element)

			output_matrix.append(row_vector)

		return output_matrix


