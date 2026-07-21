from src.data import generate_demo_dataset, validate_dataset
from src.inference import predict_record
from src.modeling import train_and_compare


def test_training_pipeline_handles_categorical_features():
    df = validate_dataset(generate_demo_dataset(600, 7))
    results = train_and_compare(df, test_size=0.2, random_state=7)
    assert len(results) == 3
    assert all(0 <= r.metrics["f1"] <= 1 for r in results)


def test_prediction_contract():
    df = validate_dataset(generate_demo_dataset(500, 9))
    model = train_and_compare(df, random_state=9)[0].pipeline
    record = df.drop(columns="label").iloc[0].to_dict()
    result = predict_record(model, record)
    assert result["prediction"] in {"normal", "attack"}
    assert 0 <= result["attack_probability"] <= 1
