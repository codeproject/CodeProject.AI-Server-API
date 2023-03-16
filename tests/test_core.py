import codeprojectai.core as cpai
import requests
import requests_mock
import pytest

MOCK_IP_ADDRESS = "localhost"
MOCK_PORT = 32168
MOCK_CUSTOM_MODEL = "mask"
OBJ_URL = "http://localhost:32168/v1/vision/detection"
OBJ_CUSTOM_URL = "http://localhost:32168/v1/vision/custom/mask"
SCENE_URL = "http://localhost:32168/v1/vision/scene"
FACE_DETECTION_URL = "http://localhost:32168/v1/vision/face"

CONFIDENCE_THRESHOLD = 0.7

MOCK_BYTES = b"Test"
MOCK_API_KEY = "mock_api_key"
MOCK_TIMEOUT = 8

MOCK_SCENE_RESPONSE = {"success": True, "label": "street", "confidence": 0.86745402}
MOCK_SCENE_PREDICTION = {"label": "street", "confidence": 0.86745402}

MOCK_OBJECT_DETECTION_RESPONSE = {
    "success": True,
    "predictions": [
        {
            "confidence": 0.6998661,
            "label": "person",
            "y_min": 0,
            "x_min": 258,
            "y_max": 676,
            "x_max": 485,
        },
        {
            "confidence": 0.7996547,
            "label": "person",
            "y_min": 0,
            "x_min": 405,
            "y_max": 652,
            "x_max": 639,
        },
        {
            "confidence": 0.59745613,
            "label": "dog",
            "y_min": 311,
            "x_min": 624,
            "y_max": 591,
            "x_max": 825,
        },
    ],
}

MOCK_OBJECT_PREDICTIONS = MOCK_OBJECT_DETECTION_RESPONSE["predictions"]
MOCK_OBJECT_CONFIDENCES = [0.6998661, 0.7996547]


MOCK_FACE_RECOGNITION_RESPONSE = {
    "success": True,
    "predictions": [
        {
            "confidence": 0.74999994,
            "userid": "Idris Elba",
            "y_min": 176,
            "x_min": 209,
            "y_max": 825,
            "x_max": 677,
        },
        {
            "confidence": 0,
            "userid": "unknown",
            "y_min": 230,
            "x_min": 867,
            "y_max": 729,
            "x_max": 1199,
        },
    ],
}

MOCK_FACE_DETECTION_RESPONSE = {
    "success": True,
    "predictions": [
        {
            "confidence": 0.9999999,
            "y_min": 173,
            "x_min": 203,
            "y_max": 834,
            "x_max": 667,
        }
    ],
}


MOCK_RECOGNIZED_FACES = {"Idris Elba": 75.0}


def test_CodeProjectAIObject_detect():
    """Test a good response from server."""
    with requests_mock.Mocker() as mock_req:
        mock_req.post(
            OBJ_URL, status_code=cpai.HTTP_OK, json=MOCK_OBJECT_DETECTION_RESPONSE
        )
        cpai_object = cpai.CodeProjectAIObject(MOCK_IP_ADDRESS, MOCK_PORT)
        predictions = cpai_object.detect(MOCK_BYTES)
        assert predictions == MOCK_OBJECT_PREDICTIONS


def test_CodeProjectAIObject_detect_custom():
    """Test a good response from server."""
    with requests_mock.Mocker() as mock_req:
        mock_req.post(
            OBJ_CUSTOM_URL, status_code=cpai.HTTP_OK, json=MOCK_OBJECT_DETECTION_RESPONSE
        )
        cpai_object = cpai.CodeProjectAIObject(
            MOCK_IP_ADDRESS, MOCK_PORT, custom_model=MOCK_CUSTOM_MODEL
        )
        predictions = cpai_object.detect(MOCK_BYTES)
        assert predictions == MOCK_OBJECT_PREDICTIONS

"""
def test_CodeProjectAIScene():
    #Test a good response from server.
    with requests_mock.Mocker() as mock_req:
        mock_req.post(SCENE_URL, status_code=cpai.HTTP_OK, json=MOCK_SCENE_RESPONSE)

        cpai_scene = cpai.CodeProjectAIScene(MOCK_IP_ADDRESS, MOCK_PORT)
        predictions = cpai_scene.recognize(MOCK_BYTES)
        assert predictions == MOCK_SCENE_PREDICTION
"""

def test_CodeProjectAIFace():
    """Test a good response from server."""
    with requests_mock.Mocker() as mock_req:
        mock_req.post(
            FACE_DETECTION_URL,
            status_code=cpai.HTTP_OK,
            json=MOCK_FACE_DETECTION_RESPONSE,
        )

        cpai_face = cpai.CodeProjectAIFace(MOCK_IP_ADDRESS, MOCK_PORT)
        predictions = cpai_face.detect(MOCK_BYTES)
        assert predictions == MOCK_FACE_DETECTION_RESPONSE["predictions"]


def test_get_objects():
    """Cant always be sure order of returned list items."""
    objects = cpai.get_objects(MOCK_OBJECT_PREDICTIONS)
    assert type(objects) is list
    assert "dog" in objects
    assert "person" in objects
    assert len(objects) == 2


def test_get_objects_summary():
    objects_summary = cpai.get_objects_summary(MOCK_OBJECT_PREDICTIONS)
    assert objects_summary == {"dog": 1, "person": 2}


def test_get_object_confidences():
    object_confidences = cpai.get_object_confidences(MOCK_OBJECT_PREDICTIONS, "person")
    assert object_confidences == MOCK_OBJECT_CONFIDENCES


def test_get_confidences_above_threshold():
    assert (
        len(
            cpai.get_confidences_above_threshold(
                MOCK_OBJECT_CONFIDENCES, CONFIDENCE_THRESHOLD
            )
        )
        == 1
    )


def test_get_recognized_faces():
    predictions = MOCK_FACE_RECOGNITION_RESPONSE["predictions"]
    assert cpai.get_recognized_faces(predictions) == MOCK_RECOGNIZED_FACES
