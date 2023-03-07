import NeuroPy as npy 
import matplotlib.pyplot as plt 




class sampleModel():
	def __init__(self):

		self.learning_rate = -0.01

		self.il_w = npy.WeightInitializer().intializeWeight((10, 3))
		self.il_b = npy.WeightInitializer().intializeWeight((1, 10))

		self.hl_w = npy.WeightInitializer().intializeWeight((2, 10))
		self.hl_b = npy.WeightInitializer().intializeWeight((1, 2))

		self.fl_w = npy.WeightInitializer().intializeWeight((3, 2))
		self.fl_b = npy.WeightInitializer().intializeWeight((1, 3))

		self.mse_arr = []


	def fit(self, trainig_data, labeld_output):
		il_o = npy.ForwardPropagation().fowardPropagation(input_value = trainig_data, weight_value = self.il_w, bias_weight = self.il_b, activation_function = "sigmoid")
		hl_o = npy.ForwardPropagation().fowardPropagation(input_value = il_o, weight_value = self.hl_w, bias_weight = self.hl_b, activation_function = "sigmoid")
		fl_o = npy.ForwardPropagation().fowardPropagation(input_value = hl_o, weight_value = self.fl_w, bias_weight = self.fl_b, activation_function = "sigmoid")

		# calculate the strenght of the final layer
		bp = npy.BackPropagation(learning_rate = self.learning_rate)

		#alpha, fwd_l_delta, prev_l_output, init_weight, activation_function_method
		fl_h = bp.getFinalLayerDelta(predicted_ouputs_vector = fl_o, actual_label_vector = labeld_output)
		fl_wd = bp.updateLayerWeight(alpha = self.learning_rate, fwd_l_delta = fl_h, prev_l_output = hl_o, init_weight = self.fl_w, activation_function_method = "sigmoid")
		self.fl_w = fl_wd
		fl_bd = bp.adjustBiasWeight(neuron_strnght = fl_h)
		self.fl_b = fl_bd


		hl_h = bp.getHiddenLayerDelta(l_output = hl_o, weight = self.fl_w, fwd_l_delta = fl_h, activation_function = "sigmoid")
		hl_wd = bp.updateLayerWeight(alpha = self.learning_rate, fwd_l_delta = hl_h, prev_l_output = il_o, init_weight = self.hl_w, activation_function_method = "sigmoid")
		self.hl_w = hl_wd
		hl_bd = bp.adjustBiasWeight(neuron_strnght = hl_h)
		self.hl_b =hl_bd


		il_h = bp.getHiddenLayerDelta(l_output = il_o, weight = self.hl_w, fwd_l_delta = hl_h, activation_function = "sigmoid")
		il_wd = bp.updateLayerWeight(alpha = self.learning_rate, fwd_l_delta = il_h, prev_l_output = trainig_data, init_weight = self.il_w, activation_function_method = "sigmoid")
		self.il_w = il_wd
		il_bd = bp.adjustBiasWeight(neuron_strnght = il_h)
		self.il_b = il_bd
		

		mse = bp.getMeanSquaredError(fl_o, labeld_output)
		self.mse_arr.append(mse)





train_data_01 = [0.95, 0.34, 0.54]
answer_01 = [1,0,0]
train_data_02 = [0.32, 0.92, 0.21]
answer_02 = [0, 1, 0]
train_data_03 = [0.4, 0.75, 0.89]
answer_03 = [0, 0, 1]

training_data = [train_data_01, train_data_02, train_data_03]
answer = [answer_01, answer_02, answer_03]


epoch = 45
model = sampleModel()

for _ in range(epoch):
	for data_index in range(len(training_data)):
		model.fit(training_data[data_index], answer[data_index])


x = [i for i in range(len(model.mse_arr))]
y = model.mse_arr
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()

print(model.fl_w)
