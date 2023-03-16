"""
CodeProject.AI SDK core.
"""
import requests
from PIL import Image
from typing import Union, List, Set, Dict

from requests.models import Response

## Const
DEFAULT_TIMEOUT        = 10  # seconds
DEFAULT_IP             = "localhost"
DEFAULT_PORT           = 32168
DEFAULT_MIN_CONFIDENCE = 0.45

## HTTP codes
HTTP_OK = 200
BAD_URL = 404

## API urls
URL_BASE_VISION      = "http://{ip}:{port}/v1/vision"
URL_CUSTOM           = "/custom/{custom_model}"
URL_OBJECT_DETECTION = "/detection"
URL_FACE_DETECTION   = "/face"
URL_FACE_REGISTER    = "/face/register"
URL_FACE_RECOGNIZE   = "/face/recognize"
URL_FACE_LIST        = "/face/list"
# URL_SCENE_RECOGNIZE = "/scene"


class CodeProjectAIException(Exception):
    pass


def format_confidence(confidence: Union[str, float]) -> float:
    """
    Takes a confidence from the API like
    0.55623 and returns 55.6 (%).
    """
    DECIMALS = 1
    return round(float(confidence) * 100, DECIMALS)


def get_confidences_above_threshold(
    confidences: List[float], confidence_threshold: float
) -> List[float]:
    """Takes a list of confidences and returns those above a confidence_threshold."""
    return [val for val in confidences if val >= confidence_threshold]


def get_recognized_faces(predictions: List[Dict]) -> List[Dict]:
    """
    Get the recognized faces.
    """
    try:
        matched_faces = {
            face["userid"]: round(face["confidence"] * 100, 1)
            for face in predictions
            if not face["userid"] == "unknown"
        }
        return matched_faces
    except:
        return {}


def get_objects(predictions: List[Dict]) -> List[str]:
    """
    Get a list of the unique objects predicted.
    """
    labels = [pred["label"] for pred in predictions]
    return sorted(list(set(labels)))


def get_object_confidences(predictions: List[Dict], target_object: str) -> List[float]:
    """
    Return the list of confidences of instances of target label.
    """
    confidences = [
        float(pred["confidence"])
        for pred in predictions
        if pred["label"] == target_object
    ]
    return confidences


def get_objects_summary(predictions: List[Dict]):
    """
    Get a summary of the objects detected.
    """
    objects = get_objects(predictions)
    return {
        target_object: len(get_object_confidences(predictions, target_object))
        for target_object in objects
    }


def post_image(
    url: str, image_bytes: bytes, timeout: int, data: dict) -> requests.models.Response:

    """Post an image to CodeProject.AI Server. Only handles exceptions."""
    
    try:
        return requests.post(url, files={"image": image_bytes}, data=data, timeout=timeout)
    except requests.exceptions.Timeout:
        raise CodeProjectAIException(f"CodeProject.AI Server connection timeout. Current " +
                                      "timeout is {timeout} seconds, try increasing this")
    except requests.exceptions.ConnectionError or requests.exceptions.MissingSchema as exc:
        raise CodeProjectAIException(f"CodeProject.AI Server connection error, check your IP and port: {exc}")


def process_image(
    url: str,
    image_bytes: bytes,
    min_confidence: float,
    timeout: int,
    data: dict = {},
) -> Dict:
    """Process image_bytes and detect. Handles common status codes"""

    data["min_confidence"] = min_confidence

    response = post_image(url=url, image_bytes=image_bytes, timeout=timeout, data=data)

    if response.status_code == HTTP_OK:
        return response.json()
    
    if response.status_code == BAD_URL:
        raise CodeProjectAIException(f"Bad url supplied, url {url} raised error {BAD_URL}")
    else:
        raise CodeProjectAIException(f"CodeProject.AI Server error: {response.status_code}")


def get_stored_faces(url, timeout) -> List:
    """Posts a request and get the stored faces as a list"""
    try:
        data = requests.post(url, timeout=timeout, data={})
    except requests.exceptions.Timeout:
        raise CodeProjectAIException(f"CodeProject.AI Server connection timeout. Current " +
                                      "timeout is {timeout} seconds, try increasing this")
    except requests.exceptions.ConnectionError or requests.exceptions.MissingSchema as exc:
        raise CodeProjectAIException(f"CodeProject.AI Server connection error, check your IP and port: {exc}")

    return data.json()


class CodeProjectAIVision:
    """Base class for CodeProject.AI vision."""

    def __init__(
        self,
        ip: str               = DEFAULT_IP,
        port: int             = DEFAULT_PORT,
        timeout: int          = DEFAULT_TIMEOUT,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        url_detect: str       = "",
        url_recognize: str    = "",
        url_register: str     = "",
        url_face_list: str    = "",
    ):
        self.port            = port
        self.timeout         = timeout
        self.min_confidence  = min_confidence

        self._url_base       = URL_BASE_VISION.format(ip=ip, port=port)
        self._url_detect     = self._url_base + url_detect
        self._url_recognize  = self._url_base + url_recognize
        self._url_register   = self._url_base + url_register
        self._url_face_list  = self._url_base + url_face_list

    def detect(self):
        """Process image_bytes and detect."""
        raise NotImplementedError

    def recognize(self):
        """Process image_bytes and recognize."""
        raise NotImplementedError

    def register(self):
        """Perform a registration."""
        raise NotImplementedError


class CodeProjectAIObject(CodeProjectAIVision):
    """Work with objects"""

    def __init__(
        self,
        ip: str               = DEFAULT_IP,
        port: int             = DEFAULT_PORT,
        timeout: int          = DEFAULT_TIMEOUT,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        custom_model: str     = "",
    ):
        if custom_model:
            url_detect = URL_CUSTOM.format(custom_model=custom_model)
        else:
            url_detect = URL_OBJECT_DETECTION
            
        super().__init__(
            ip             = ip,
            port           = port,
            timeout        = timeout,
            min_confidence = min_confidence,
            url_detect     = url_detect,
        )

    def detect(self, image_bytes: bytes):
        """Process image_bytes and detect."""
        response = process_image(
            url            = self._url_detect,
            image_bytes    = image_bytes,
            min_confidence = self.min_confidence,
            timeout        = self.timeout,
        )
        return response["predictions"]

"""
class CodeProjectAIScene(CodeProjectAIVision):
    # Work with scenes

    def __init__(
        self,
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        timeout: int = DEFAULT_TIMEOUT,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        super().__init__(
            ip=ip,
            port=port,
            timeout=timeout,
            min_confidence=min_confidence,
            url_recognize=URL_SCENE_RECOGNIZE,
        )

    def recognize(self, image_bytes: bytes):
        #Process image_bytes and detect.
        response = process_image(
            url=self._url_recognize,
            image_bytes=image_bytes,
            min_confidence=self.min_confidence,
            timeout=self.timeout,
        )
        del response["success"]
        return response
"""

class CodeProjectAIFace(CodeProjectAIVision):
    """Work with objects"""

    def __init__(
        self,
        ip: str               = DEFAULT_IP,
        port: int             = DEFAULT_PORT,
        timeout: int          = DEFAULT_TIMEOUT,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        super().__init__(
            ip             = ip,
            port           = port,
            timeout        = timeout,
            min_confidence = min_confidence,
            url_detect     = URL_FACE_DETECTION,
            url_register   = URL_FACE_REGISTER,
            url_recognize  = URL_FACE_RECOGNIZE,
            url_face_list  = URL_FACE_LIST,
        )

    def detect(self, image_bytes: bytes):
        """Process image_bytes and detect."""
        response = process_image(
            url            = self._url_detect,
            image_bytes    = image_bytes,
            min_confidence = self.min_confidence,
            timeout        = self.timeout,
        )
        return response["predictions"]

    def register(self, name: str, image_bytes: bytes):
        """
        Register a face name to a file.
        """
        response = process_image(
            url            = self._url_register,
            image_bytes    = image_bytes,
            min_confidence = self.min_confidence,
            timeout        = self.timeout,
            data           = { "userid": name },
        )

        if response["success"] == True:
            return response["message"]

        elif response["success"] == False:
            error = response["error"]
            raise CodeProjectAIException(
                f"CodeProject.AI Server raised an error registering a face: {error}"
            )

    def recognize(self, image_bytes: bytes):
        """Process image_bytes, performing recognition."""
        response = process_image(
            url            = self._url_recognize,
            image_bytes    = image_bytes,
            min_confidence = self.min_confidence,
            timeout        = self.timeout,
        )

        return response["predictions"]

    def get_registered_faces(self):
        """Get the name of the registered faces"""
        response = get_stored_faces(
            url=self._url_face_list, timeout=self.timeout
        )
        return response["faces"]
