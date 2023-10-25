from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 12

@test_case(points=4.0, hidden=False)
def test_q5_1_public_1(np, pd, sklearn, data_splits):
    features_final = pd.read_csv("./data/q5_1_public_1_data_X.csv")
    income = pd.read_csv("./data/q5_1_public_1_data_y.csv")

    expected_X_train = pd.read_csv("./data/q5_1_public_1_solution_X_train.csv")
    expected_X_test = pd.read_csv("./data/q5_1_public_1_solution_X_test.csv")
    expected_y_train = pd.read_csv("./data/q5_1_public_1_solution_y_train.csv")
    expected_y_test = pd.read_csv("./data/q5_1_public_1_solution_y_test.csv")
    
    X_train, X_test, y_train, y_test = data_splits(features_final, income)

    np.testing.assert_allclose(X_train, expected_X_train)
    np.testing.assert_allclose(X_test, expected_X_test)
    np.testing.assert_allclose(y_train, expected_y_train)
    np.testing.assert_allclose(y_test, expected_y_test)

