from src.ml.nlp.pipline import TrafficPipeline # type: ignore


def main() -> None:
    pipeline = TrafficPipeline()

    sample_structured_data = {
        "Hour": 8,
        "Month": 12,
        "Weekday": 2,
        "Temperature(F)": 45.0,
        "Humidity(%)": 70.0,
        "Visibility(mi)": 8.0,
        "Wind_Speed(mph)": 5.0,
        "Amenity": 0,
        "Bump": 0,
        "Crossing": 1,
        "Give_Way": 0,
        "Junction": 1,
        "No_Exit": 0,
        "Railway": 0,
        "Roundabout": 0,
        "Station": 0,
        "Stop": 0,
        "Traffic_Signal": 1,
    }

    sample_description = "accident on right lane with moderate traffic congestion"

    prediction = pipeline.predict(sample_structured_data, sample_description)
    print(f"Predicted Severity: {prediction}")


if __name__ == "__main__":
    main()
