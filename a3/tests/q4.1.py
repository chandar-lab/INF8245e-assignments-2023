from otter.test_files import test_case

OK_FORMAT = False

name = "q4.1"
points = 5

@test_case(points=1.5, hidden=False)
def test_q4_1_public_1(np, pd, one_hot_encoding):
    features_log_minmax_transform = pd.read_csv("./data/q3_1_public_1.csv")
    
    expected_features_final = pd.read_csv("./data/q4_1_public_1.csv")

    features_final = one_hot_encoding(features_log_minmax_transform)

    features_final = np.array(features_final, dtype=float)
    expected_features_final = np.array(expected_features_final, dtype=float)

    np.testing.assert_allclose(features_final, expected_features_final)

