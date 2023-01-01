from app import form_response 


input_data = {
    "correct_values":
    {"SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2,
    }

}

# Test the form_response function
def test_form_response_incorrect_values(data=input_data["correct_values"]):
    # Test that the form_response function returns a string
    res=form_response(data)
    # Test that the form_response function returns a string
    assert res in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]