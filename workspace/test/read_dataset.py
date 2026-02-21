import pickle

data_path = "my_demo_session/dataset_plan.pkl"
with open(data_path, 'rb') as f:
	dataset = pickle.load(f)

print("Dataset keys:", dataset.keys())