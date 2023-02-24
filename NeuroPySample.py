import NeuroPy as npy
import matplotlib.pyplot as plt



sample_input = [npy.WeightInitializationMethods().radomInitializer() for i in range(20)]
correct_answer = [0, 1]


l1_weights = npy.WeightInitializer().initNormalizedXavierWeight([6, 20], 6, 4)
l1_biased_weights = npy.WeightInitializer().initNormalizedXavierWeight([1, 6], 6, 4)

l2_weights = npy.WeightInitializer().initNormalizedXavierWeight([4, 6], 4, 2)
l2_biased_weights = npy.WeightInitializer().initNormalizedXavierWeight([1, 4], 6, 4)

fl_weights = npy.WeightInitializer().intializeWeight([2, 4])
f1_biased_weights = npy.WeightInitializer().initNormalizedXavierWeight([1, 2], 6, 4)


count = 0



epoch = 100
erro_updates = []

for i in range(epoch):
    l1_output = npy.ForwardPropagation().createLayer(sample_input, l1_weights, l1_biased_weights)
    l2_output = npy.ForwardPropagation().createLayer(l1_output, l2_weights, l2_biased_weights)
    fl_output = npy.ForwardPropagation().createLayer(l2_output, fl_weights, f1_biased_weights)


    #print("Prediction: ", npy.ActivationFunction().argMAx(fl_output), " Correct answer: ", correct_answer)
    error_calc = npy.Backpropagation().getMeanSquaredError(fl_output, correct_answer)
    #print("Count: ", count, "Error calculation: ", error_calc,  " \n \n ")
    erro_updates.append(error_calc)


    back_propagation = npy.Backpropagation(learning_rate = -0.5)

    fl_bp_neuron_strenght = back_propagation.getFLayerNeuronStrenght(fl_output, correct_answer)


    fl_calculated_weight_adjustments = back_propagation.calculateWeightAdjustment(fl_bp_neuron_strenght, l2_output)
    fl_applied_weight_adjustment = back_propagation.applyWeightAdjustment(fl_weights, fl_calculated_weight_adjustments)
    l2_layer_neuron_strenght = back_propagation.getHLayerNeuronStrength(l2_output, fl_applied_weight_adjustment, fl_bp_neuron_strenght)


    l2_calculated_weight_adjustments = back_propagation.calculateWeightAdjustment(l2_layer_neuron_strenght, l1_output)
    l2_applied_weight_adjustment = back_propagation.applyWeightAdjustment(l2_weights, l2_calculated_weight_adjustments)
    l1_layer_neuron_strenght = back_propagation.getHLayerNeuronStrength(l1_output, l2_applied_weight_adjustment, l2_layer_neuron_strenght)


    l3_calculated_weight_adjustments = back_propagation.calculateWeightAdjustment(l1_layer_neuron_strenght, sample_input)
    l3_applied_weight_adjustment = back_propagation.applyWeightAdjustment(l1_weights, l3_calculated_weight_adjustments)


    #back_propagation.getAdjustedBiasdWeights(fl_bp_neuron_strenght)
    f1_biased_weights = back_propagation.getAdjustedBiasdWeights(fl_bp_neuron_strenght)
    l2_biased_weights = back_propagation.getAdjustedBiasdWeights(l2_layer_neuron_strenght)
    l1_biased_weights = back_propagation.getAdjustedBiasdWeights(l1_layer_neuron_strenght)
    #print(back_propagation.getAdjustedBiasdWeights(l1_layer_neuron_strenght))


    fl_weights = fl_applied_weight_adjustment
    l2_weights = l2_applied_weight_adjustment
    l1_output = l3_applied_weight_adjustment
    count += 1



x = [i for i in range(len(erro_updates))]
y = erro_updates
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()