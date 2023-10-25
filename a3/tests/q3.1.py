from otter.test_files import test_case

OK_FORMAT = False

name = "q3.1"
points = 7

@test_case(points=2.5, hidden=False)
def test_q3_1_public_1(np, pd, sklearn, normalize_features):
    data = pd.read_csv("./data/census.csv")
    features_raw = data.drop('income', axis = 1)
    
    features_log_minmax_transform = normalize_features(features_raw)
    
    expected_features_log_minmax_transform = pd.read_csv("./data/q3_1_public_1.csv")
    
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    np.testing.assert_allclose(features_log_minmax_transform[numerical], expected_features_log_minmax_transform[numerical])

