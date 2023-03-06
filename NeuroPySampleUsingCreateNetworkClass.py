import NeuroPy as npy
import pprint

pp = pprint.PrettyPrinter(width=41, compact=True)

hidden_layers = [4, 2, 2]

model = npy.CreateNetwork(
	input_size = 3, 
	layer_size_vectors = hidden_layers, 
	learning_rate = -0.01, 
	weight_initializer = "simple", 
	regularization_method = "none", 
	l2_penalty = 0.01
	)

train_data_01 = [1, 1, 1]
answer_01 = [1,0]

train_data_02 = [1, 0, 1]
answer_02 = [0, 1]

train_data_03 = [0, 0, 0]
answer_03 = [1, 0]


training_data = [train_data_01, train_data_02, train_data_03]
answer = [answer_01, answer_02, answer_03]

# fit the data
result = model.fit(training_data, answer, 3000, 1)
print("prediction: ", model.predict(train_data_02))
pp.pprint(model.weights_set)


import matplotlib.pyplot as plt 
x = [i for i in range(len(model.mean_square_error_log))]
y = model.mean_square_error_log
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()
