from otter.test_files import test_case

OK_FORMAT = False

name = "q4.2"
points = 5

@test_case(points=1.5, hidden=False)
def test_q4_1_public_2(np, pd, encode_label):
    data = pd.read_csv("./data/census.csv")
    income_raw = data['income']
    
    income = encode_label(income_raw)

    expected_income = pd.read_csv("./data/q4_1_public_2.csv")
    
    assert income.equals(expected_income['income'])

