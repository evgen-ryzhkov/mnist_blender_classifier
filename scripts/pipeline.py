from scripts.data import MNISTData
from scripts.model import DigitClassifier


#  --------------
# Step 1
# getting data and familiarity with it

data_obj = MNISTData()
X_train, y_train = data_obj.get_train_set()


#  --------------
# Step 2
# training level 1 models
clf_o = DigitClassifier()
# clf_o.train_1st_level_models(X_train, y_train)


#  --------------
# Step 3
# getting valid and test data sets
X_valid, y_valid, X_test, y_test = data_obj.get_validate_and_test_sets()


#  --------------
# Step 4
# training level 2 model (blender)
#clf_o.train_2st_level_model(X_valid, y_valid)


#  --------------
# Step 5
# comparing individual models and blender
clf_o.compare_individual_models_and_blender(X_test, y_test)
