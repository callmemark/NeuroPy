import NeuroPy as npy
import matplotlib.pyplot as plt

# the amount of elements inside the list represent the aount of hidden layers plus the final element as the final layer
# each value represent the number pf neurons in that layer
hidden_layers = [6, 4, 4, 3, 10, 12, 5, 3]

# The first parameter of the class is the amount of input, while the second one are the hidden layers
networks = npy.CreateNetwork(5, hidden_layers)

train_data_01 = [0.68, 0.95, 0.12, 0.1, 0.1]
answer_01 = [1,0,0]
train_data_02 = [0.25, 0.65, 0.72, 0.1, 0.1]
answer_02 = [0, 1, 0]
train_data_03 = [0.91, 0.02, 0.12, 0.1, 0.1]
answer_03 = [0, 0, 1]

training_data = [train_data_01, train_data_02, train_data_03]
answer = [answer_01, answer_02, answer_03]

# fit the data
result = networks.fit(training_data, answer, 4, 2)

# plot the mean square error
x = [i for i in range(len(networks.mean_square_error_log))]
y = networks.mean_square_error_log
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()