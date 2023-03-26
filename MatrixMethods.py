"""
Author: MArk John Velmonte

Description: 

	Create new object 'matrix' for matrix operations
"""

from VectorMethods import Vector
from math import isnan, log

class Matrix():
	def __init__(self):
		pass


	def isMatrixValid(self, matrix):
		"""
			Check if the matrix is valid meaning it have equal row and col and earch value homogeneous
		"""

		if self.isSquare(matrix) and self.isHomogenous(matrix) and self.notNaN(matrix) and self.noInf(matrix):
			return True
		else:
			return False


	def noInf(self, matrix):
		"""
			look at the elemtns of matrix to check if there is a infinite value
		"""
		returned_value = True
		for vect_index in range(len(matrix)):
			if Vector(matrix[vect_index]).noInf() == False:
				returned_value = False
				break

		return returned_value




	def isSquare(self, matrix):
	    """
	    Checks if a 2D matrix has equal row and column sizes.
	    Returns:
	        bool: True if matrix has equal row and column sizes, False otherwise
	    """
	    if not isinstance(matrix, list):
	    	return False
	    	
	    num_rows = len(matrix)
	    if num_rows == 0:
	        return True
	    num_cols = len(matrix[0])
	    for row in matrix:
	        if len(row) != num_cols:
	            return False
	    return True


	def isHomogenous(self, matrix):
	    """
	    Check if a multidimensional array is homogenous (all elements have the same type).
	    Returns:
	    	True if the array is homogenous, False otherwise.
	    """
	    if not isinstance(matrix, list):
	        return True  
	    
	    if len(matrix) == 0:
	        return True 

	    first_type = type(matrix[0])
	    for element in matrix:
	        if type(element) != first_type:
	            return False
	    
	    for sub_arr in matrix:
	        if not self.isHomogenous(sub_arr):
	            return False
	    
	    return True


	def matrixDotProd(self, matrix_01, matrix_02):
		"""
		Calcualate the dot product of matrix_01 argument with respect to each vector in the matrix_02 argument
		Return matrix containing vector dot procuct
		"""

		output_matrix = []

		for mtx_1_vec in matrix_01:
			dprod_row = []
			for mtx_2_T_vec in matrix_02:
				dprod = Vector(mtx_1_vec).dotProd(mtx_2_T_vec)

				dprod_row.append(dprod)
			output_matrix.append(dprod_row)

		return output_matrix



	def matrixVectorDotProd(self, matrix, vector):
		"""
			Calculate the dot product of a vector and a matrix

			Parameters:
				vector (vector): A 1-dimensional "list" representing a vector.
				matrix (matrix): A 2-dimensional "list" representing a matrix.

			Returns (Vector) A vector representing the result of the dot product.
		"""
		if len(vector) != len(matrix[0]):
			raise ValueError("The dimensions of the vector and matrix do not match.")

		output_vector = []
		for mtx_vec in matrix:
			d_prod = 0
			for vec_val, mtx_vec_val in zip(vector, mtx_vec):
				d_prod += vec_val * mtx_vec_val
			output_vector.append(d_prod)

		return Vector(output_vector)



	def outerProduct(self, multiplicand_vector, multiplier_vector):
		"""
			NOTE: outerProduct or tensor product is technically a tensor operation but since this is the only tensor operation i use it will be under matrix
			calculate the outer product of the two vector 
			
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



	def matrixVectorMultiply(self, multiplicand_matrix, multiplicand_vector):
		"""
			multiply matrix to a vector
			
			Argumemts:
			multiplicand_matrix	(Matrix) 
			multiplicand_vector	(Vector)

			Return (Matrix) A two dimensional matrix 
		"""
		output_matrix = []

		for mtx_vec_row in multiplicand_matrix:
			row = []
			for mtx_vec_row_val, vec_val in zip(mtx_vec_row, multiplicand_vector):
				row.append(mtx_vec_row_val * vec_val)
			output_matrix.append(row)

		return output_matrix


	def getSumOfRow(self, matrix):
		"""
			caculate the sum of rows of a matrix array
			
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


	def matrixAddition(self, matrix_01, matrix_02):
		"""
			add two matrix
			
			Argumemts:
			matrix_01	(2d matrix Vector)
			matrix_02	(2d matrix Vector)

			Return 2d Matrix Vector
		"""

		mtx_01_shape = len(matrix_01)
		mtx_02_shape = len(matrix_02)

		if mtx_01_shape == mtx_02_shape:
			output_matrix = [[0 for _ in range(len(matrix_01[0]))] for _ in range(len(matrix_01))]

			for i in range(len(matrix_01)):
				for j in range(len(matrix_01[0])):
					output_matrix[i][j] = matrix_01[i][j] + matrix_02[i][j]

		elif self.getShape(matrix_01)[1] == self.getShape(matrix_02)[1]:
			output_matrix = []
			for mtx_01_vec in matrix_01:
				for mtx_02_vec in matrix_02:
					output_matrix.append(Vector(mtx_01_vec).addVector(mtx_02_vec))
		else:
			err_msg = "Erro matrix have no common shapes row != row and col != col"
			raise ValueError(err_msg)

		return output_matrix



	def matrixMultiply(self, matrix_01, matrix_02):
		"""
			add two matrix
			
			Argumemts:
			matrix_01	(2d matrix Vector)
			matrix_02	(2d matrix Vector)

			Return 2d Matrix Vector
		"""

		if len(matrix_01) != len(matrix_02) or len(matrix_01[0]) != len(matrix_02[0]):
		    raise ValueError("Arrays must have the same shape")

		result = [[0 for _ in range(len(matrix_01[0]))] for _ in range(len(matrix_01))]

		for i in range(len(matrix_01)):
			for j in range(len(matrix_01[0])):
				result[i][j] = matrix_01[i][j] * matrix_02[i][j]

		return result



	def matrixSubtract(self, matrix_minuend, matrix_subtrahend):
		"""
			subtract two matrix
			
			Argumemts:
			matrix_minuend	(2d matrix Vector)
			matrix_subtrahend	(2d matrix Vector)

			Return 2d Matrix Vector
		"""
		if len(matrix_minuend) != len(matrix_subtrahend) or len(matrix_minuend[0]) != len(matrix_subtrahend[0]):
			raise ValueError("Arrays must have the same shape")

		result = [[0 for _ in range(len(matrix_minuend[0]))] for _ in range(len(matrix_minuend))]

		for i in range(len(matrix_minuend)):
			for j in range(len(matrix_minuend[0])):
				result[i][j] = matrix_minuend[i][j] - matrix_subtrahend[i][j]

		return result


	def flatten(self, matrix_arr):
		"""
			transform a matrix array into a 1d vector array
			
			Argumemts:
			matrix_arr	(nd vector Vector)

			Return 1d vector Vector
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


	def getShape(self, matrix):
		"""
			get the shape of vector / matrix array
			
			Argumemts:
			matrix	(nd Vector)

			Return 1d vector Vector
		"""
		if self.isHomogenous(matrix):
			shape = []

			while isinstance(matrix, list):
				shape.append(len(matrix))
				matrix = matrix[0]
		else:
			err_msg = "Cant get shape of a matrix that is not homogenous"
			raise ValueError(err_msg)

		return shape


	def transpose(self, arr_arg):
		""""
			Transpose the given 2d matrix array swapping its elemtns positions

			Arguements			:	arr_arg
			Returns(Vector) 		:	matrix array
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
			Returns(Vector) 		:	matrix array
		"""
		squared_arr = []
		arr_shape = self.getShape(vector_array)
		if len(arr_shape) != 1:
			raise Exception("Error in function vectorSquare, Vector should be one dimesion but have " + str(arr_shape))

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

		return output_array



	def matrixScalarDevision(self, matrix, scalar_devider):
		output_array = []

		for row in matrix:
			col_val_sum = []
			for col_val in row:
				col_val_sum.append(col_val / scalar_devider)

			output_array.append(col_val_sum)

		return output_array



	def matrixScalarSubtract(self, matrix, scalar_minuend):
		output_array = []

		for row in matrix:
			col_val_sum = []
			for col_val in row:
				col_val_sum.append(col_val - scalar_minuend)

			output_array.append(col_val_sum)

		return output_array



	def matrixScalarAddition(self, matrix, scalar_minuend):
		output_array = []

		for row in matrix:
			col_val_sum = []
			for col_val in row:
				col_val_sum.append(col_val + scalar_minuend)

			output_array.append(col_val_sum)

		return output_array



	def matrixAverage(self, ndMatrix):
		"""
			Calculate the matrix indide the list of matrix
			
			Arguments:
			ndMatrix (Matrix)				: A list holding multiple matrix
		"""

		matrix_count = len(ndMatrix)
		ind_matrix_count = len(ndMatrix[0])

		ouput_matrix = []

		for mtx in ndMatrix:
			if len(ouput_matrix) == 0:
				ouput_matrix = mtx
				continue

			n_vec = 0
			for vector_index in range(len(mtx)):
				n_vec += 1
				ouput_matrix[vector_index] = Vector(ouput_matrix[vector_index]).addVector(mtx[vector_index])

			n_vec = 0

		ouput_matrix = self.matrixScalarDevision(ouput_matrix, matrix_count)

		return ouput_matrix




	def matrixSum(self, matrix):
		"""
			Return (Scalar)
		"""
		vector_sum = 0
		for vector in matrix:
			vector_sum += Vector(vector).sum()

		return vector_sum



	def rowAverage(self, matrix):
		"""
		Return matrix of the sum of each row
		"""
		output_matrix = []

		for mtx_row in matrix:
			output_matrix.append([Vector(mtx_row).sum() / len(mtx_row)])

		return output_matrix


	def columnAverage(self, matrix):
		"""
			Return the average of the column 
		"""
		return self.transpose(self.rowAverage(self.transpose(matrix)))


	def clip(self, matrix, min_arg, max_arg):
		"""
			Clip the vectors in matrix to given min and max value
		"""
		matrix_output = []

		for vect_index in range(len(matrix)):
			#print(matrix[vect_index])
			matrix_output.append(Vector(matrix[vect_index]).clip(min_arg, max_arg))

		return matrix_output


	def notNaN(self, matrix):
		"""
			Check if ocntains a NaN value
		"""
		returned_value = True

		for vect_index in range(len(matrix)):
			if Vector(matrix[vect_index]).notNaN() == False:
				returned_value = False
				break

		return returned_value


	def log(self, matrix, direction = "+"):
		"""
			get the log of elemetments in array
		"""
		output_matrix = []

		for vec_index in range(len(matrix)):
			row_vect = []
			for elem_index  in range(len(matrix[vec_index])):
				if direction == "+":
					row_vect.append(log(matrix[vec_index][elem_index]))
				elif direction == "-":
					row_vect.append(-log(matrix[vec_index][elem_index]))
				else:
					err_msg = "No operation found " + direction
					raise ValueError(err_msg)

			output_matrix.append(row_vect)

		return output_matrix




class Tensor3D():
	def __init__(self):
		#super().__init__()
		pass

	def columnAverage(self, tensor_3d):
		if Matrix().isHomogenous(tensor_3d) == False:
			err_msg = "tensor should be homogenous"
			raise ValueError(err_msg)

		summed_output_matrix = tensor_3d[0]
		grouped_matrix = [[] for _ in range(len(tensor_3d[0]))]

		for out_row_index in range(len(tensor_3d)):
			for col_index in range(len(tensor_3d[out_row_index])):
				grouped_matrix[col_index].append(tensor_3d[out_row_index][col_index])
				

		output_matrix = []
		for value in grouped_matrix:
			output_matrix.append(Matrix().columnAverage(value)[0])


		return output_matrix
		
