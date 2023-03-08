import NeuroPy as npy




saved_model = npy.LoadModel("SampleSet.json")



train_data_01 = [0.95, 0.34, 0.54]
answer_01 = [1,0,0]
train_data_02 = [0.32, 0.92, 0.21]
answer_02 = [0, 1, 0]
train_data_03 = [0.4, 0.75, 0.89]
answer_03 = [0, 0, 1]

training_data = [train_data_01, train_data_02, train_data_03]
answer = [answer_01, answer_02, answer_03]


# fit the data
saved_model.fit(training_data, answer, 100, 1)
saved_model.saveModelToJson("SampleSet")