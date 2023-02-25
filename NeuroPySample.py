"""

  It is ideal to transfer the code to jypyter notbook or google colab to have different cells run
  since everytime the SimpleModel class is initialized its value will reset including all the trained
  weights

"""


import NeuroPy as npy
import matplotlib.pyplot as plt

class SimpleModel():
  def __init__(self):
    self.l1_weights = npy.WeightInitializer().initNormalizedXavierWeight([6, 3], 6, 4)
    self.l1_biased_weights = npy.WeightInitializer().initNormalizedXavierWeight([1, 6], 6, 4)

    self.l2_weights = npy.WeightInitializer().initNormalizedXavierWeight([4, 6], 4, 2)
    self.l2_biased_weights = npy.WeightInitializer().initNormalizedXavierWeight([1, 4], 6, 4)

    self.fl_weights = npy.WeightInitializer().intializeWeight([3, 4])
    self.f1_biased_weights = npy.WeightInitializer().initNormalizedXavierWeight([1, 3], 6, 4)

    self.learning_rate_arr = []



  def predictImage(self, image_arr):
    self.l1_output = npy.ForwardPropagation().createLayer(image_arr, self.l1_weights, self.l1_biased_weights)
    self.l2_output = npy.ForwardPropagation().createLayer(self.l1_output, self.l2_weights, self.l2_biased_weights)
    self.fl_output = npy.ForwardPropagation().createLayer(self.l2_output, self.fl_weights, self.f1_biased_weights)
    print("Prediction: ", self.fl_output)

    return self.fl_output



  def learn(self, image_arr, correct_answer):
    self.l1_output = npy.ForwardPropagation().createLayer(image_arr, self.l1_weights, self.l1_biased_weights)
    self.l2_output = npy.ForwardPropagation().createLayer(self.l1_output, self.l2_weights, self.l2_biased_weights)
    self.fl_output = npy.ForwardPropagation().createLayer(self.l2_output, self.fl_weights, self.f1_biased_weights)

    back_propagation = npy.BackPropagation(learning_rate = -0.01)

    fl_bp_neuron_strenght = back_propagation.getFLayerNeuronStrenght(self.fl_output, correct_answer)

    fl_calculated_weight_adjustments = back_propagation.calculateWeightAdjustment(fl_bp_neuron_strenght, self.l2_output)
    fl_applied_weight_adjustment = back_propagation.applyWeightAdjustment(self.fl_weights, fl_calculated_weight_adjustments)
    l2_layer_neuron_strenght = back_propagation.getHLayerNeuronStrength(self.l2_output, fl_applied_weight_adjustment, fl_bp_neuron_strenght)


    l2_calculated_weight_adjustments = back_propagation.calculateWeightAdjustment(l2_layer_neuron_strenght, self.l1_output)
    l2_applied_weight_adjustment = back_propagation.applyWeightAdjustment(self.l2_weights, l2_calculated_weight_adjustments)
    l1_layer_neuron_strenght = back_propagation.getHLayerNeuronStrength(self.l1_output, l2_applied_weight_adjustment, l2_layer_neuron_strenght)


    l3_calculated_weight_adjustments = back_propagation.calculateWeightAdjustment(l1_layer_neuron_strenght, image_arr)
    l3_applied_weight_adjustment = back_propagation.applyWeightAdjustment(self.l1_weights, l3_calculated_weight_adjustments)


    self.f1_biased_weights = back_propagation.getAdjustedBiasdWeights(fl_bp_neuron_strenght)
    self.l2_biased_weights = back_propagation.getAdjustedBiasdWeights(l2_layer_neuron_strenght)
    self.l1_biased_weights = back_propagation.getAdjustedBiasdWeights(l1_layer_neuron_strenght)
 

    self.fl_weights = fl_applied_weight_adjustment
    self.l2_weights = l2_applied_weight_adjustment
    self.l1_output = l3_applied_weight_adjustment

    self.error_calc = npy.BackPropagation().getMeanSquaredError(self.fl_output, correct_answer)
    self.learning_rate_arr.append(self.error_calc)


  def plotLearningCcurve(self):
    x = [i for i in range(len(self.learning_rate_arr))]
    y = self.learning_rate_arr
    plt.title("Mean square error") 
    plt.xlabel("Epoch") 
    plt.ylabel("Error value")
    plt.plot(x, y) 
    plt.show()



model = SimpleModel()



train_data_01 = [0.68, 0.95, 0.12]
answer_01 = [1,0,0]

train_data_02 = [0.25, 0.65, 0.72]
answer_02 = [0, 1, 0]

train_data_02 = [0.91, 0.02, 0.12]
answer_03 = [0, 0, 1]


training_data = [train_data_01, train_data_02, train_data_02]
answer = [answer_01, answer_02, answer_03]


learn_cycle = 10
epoch = 1


for i in range(epoch):
  print("epoch: ", i)
  for j in range(learn_cycle):
    for k in range(3):
      model.learn(training_data[k], answer[k])


## A slight difference to train_data_01
test_data_01 = [0.69, 0.93, 0.10]
test_asnwer_01 = [1,0,0]

test_data_02 = [0.21, 0.71, 0.72]
test_asnwer_02 = [0, 1, 0]

test_data_03 = [0.89, 0.06, 0.09]
test_asnwer_03 = [0, 0, 1]



result_01 = model.predictImage(test_data_01)
print("Prediction: ", npy.ActivationFunction().argMax(result_01), " correct_answer: ",  test_asnwer_01)
result_02 = model.predictImage(test_data_02)
print("Prediction: ", npy.ActivationFunction().argMax(result_02), " correct_answer: ",  test_asnwer_02)
result_03 = model.predictImage(test_data_03)
print("Prediction: ", npy.ActivationFunction().argMax(result_03), " correct_answer: ",  test_asnwer_03)


model.plotLearningCcurve()
