import NeuroPy as npy
#import matplotlib.pyplot as plt 


hidden_layers = [(3, "sigmoid"), (10, "sigmoid"), (12, "sigmoid"), (10, "sigmoid"), (3, "sigmoid")]

model = npy.CreateNetwork(
	input_size = 3, 
	layer_size_vectors = hidden_layers, 
	learning_rate = -0.01, 
	weight_initializer = "simple", 
	regularization_method = "none", 
	l2_penalty = 0.01
	)



train_data_01 = [0.95, 0.34, 0.54]
answer_01 = [1,0,0]
train_data_02 = [0.32, 0.92, 0.21]
answer_02 = [0, 1, 0]
train_data_03 = [0.4, 0.75, 0.89]
answer_03 = [0, 0, 1]

training_data = [train_data_01, train_data_02, train_data_03]
answer = [answer_01, answer_02, answer_03]


# fit the data
model.fit(training_data, answer, 1, 1)


"""

x = [i for i in range(len(model.mean_square_error_log))]
y = model.mean_square_error_log
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()
"""