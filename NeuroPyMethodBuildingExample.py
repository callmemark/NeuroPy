import NeuroPy as npy 
import matplotlib.pyplot as plt 




class sampleModel():
	def __init__(self):

		self.il_w = npy.WeightInitializer().intializeWeight((4, 3))
		self.il_b = npy.WeightInitializer().intializeWeight((1, 4))

		self.hl_w = npy.WeightInitializer().intializeWeight((2, 4))
		self.hl_b = npy.WeightInitializer().intializeWeight((1, 2))

		self.fl_w = npy.WeightInitializer().intializeWeight((2, 2))
		self.fl_b = npy.WeightInitializer().intializeWeight((1, 2))

		self.mse_arr = []

	def fit(self, trainig_data, labeld_output):
		il_o = npy.ForwardPropagation().fowardPropagation(input_value = trainig_data, weight_value = self.il_w, bias_weight = self.il_b)
		hl_o = npy.ForwardPropagation().fowardPropagation(input_value = il_o, weight_value = self.hl_w, bias_weight = self.hl_b)
		fl_o = npy.ForwardPropagation().fowardPropagation(input_value = hl_o, weight_value = self.fl_w, bias_weight = self.fl_b)

		# calculate the strenght of the final layer
		bp = npy.BackPropagation(learning_rate = -0.01)

		fl_h = bp.getFLayerNeuronStrenght(predicted_ouputs_vector = fl_o, actual_label_vector = labeld_output)
		fl_wd = bp.updateLayerWeight(succeeding_layer_neuron_strenght = fl_h, preceding_neuron_output = hl_o, initial_weight_matrix = self.fl_w)
		self.fl_w = fl_wd
		fl_bd = bp.getAdjustedBiasdWeights(neuron_strnght = fl_h)
		self.fl_b = fl_bd


		hl_h = bp.getHLayerNeuronStrength(preceding_neuron_output_arr = hl_o, weight = self.fl_w, proceding_neuron_output_arr = fl_h)
		hl_wd = bp.updateLayerWeight(succeeding_layer_neuron_strenght = hl_h, preceding_neuron_output = il_o, initial_weight_matrix = self.hl_w)
		self.hl_w = hl_wd
		hl_bd = bp.getAdjustedBiasdWeights(neuron_strnght = hl_h)
		self.hl_b =hl_bd


		il_h = bp.getHLayerNeuronStrength(preceding_neuron_output_arr = il_o, weight = self.hl_w, proceding_neuron_output_arr = hl_h)
		il_wd = bp.updateLayerWeight(succeeding_layer_neuron_strenght = il_h, preceding_neuron_output = trainig_data, initial_weight_matrix = self.il_w)
		self.il_w = il_wd
		il_bd = bp.getAdjustedBiasdWeights(neuron_strnght = il_h)
		self.il_b = il_bd
		
		mse = bp.getMeanSquaredError(fl_o, labeld_output)
		self.mse_arr.append(mse)





sample_data = [
				[1, 1, 1],
				[1, 0, 1],
				[0, 1, 0],
				[0, 0, 0]
			]

labeld_data = [
				[1, 0],
				[1, 0],
				[0, 1],
				[0, 1]
			]


epoch = 1000
model = sampleModel()

for _ in range(epoch):
	for data_index in range(len(sample_data)):
		model.fit(sample_data[data_index], labeld_data[data_index])


x = [i for i in range(len(model.mse_arr))]
y = model.mse_arr
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()