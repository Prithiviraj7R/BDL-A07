from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
import numpy as np
import cv2
import uvicorn
import time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import psutil

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Custom metrics
ip_counter = Gauge('api_requests_by_ip', 'Number of requests by IP address', ['ip'])
api_runtime_gauge = Gauge('api_runtime', 'API runtime in milliseconds')
api_tl_time_gauge = Gauge('api_tl_time', 'API T/L time in microseconds per character')
api_memory_usage_gauge = Gauge('api_memory_usage', 'API memory usage in bytes')
api_cpu_usage_gauge = Gauge('api_cpu_usage', 'API CPU utilization rate')
api_network_bytes_gauge = Gauge('api_network_bytes', 'API network I/O bytes')
api_network_bytes_rate_gauge = Gauge('api_network_bytes_rate', 'API network I/O bytes rate')


def format_image(image: bytes) -> np.array:
    """
    A function to process the uploaded image.
    It converts the image into grey scale.
    It resizes the image into 28*28
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    img = cv2.bitwise_not(img) 

    height, width = img.shape[:2]
    input_length = height * width

    # converting the image into an array
    data = np.array(img)
    data = data / 255.0
    data = data.reshape((1, 28, 28, 1))
    return data, input_length

def predict_digit(data: np.array) -> str:
    """
    A function to predict the digit given
    an array representing the image.
    """
    digit = str(np.random.randint(0, 10))
    return digit

@app.post('/predict/')
async def digit_classification(request: Request, file: UploadFile = File(...)):
    """
    Accepts the uploaded image and passes
    the array returned by format_image function
    to predict_digit function for prediction.
    """

    start_time = time.time()

    image = await file.read()
    data, input_length = format_image(image)
    digit = predict_digit(data)

    # ip counter for api requests
    client_ip = request.client.host
    ip_counter.labels(ip=client_ip).inc()
    
    # runtime calculation
    final_time = time.time()
    elapsed_time = (final_time - start_time) * 1000 
    api_runtime_gauge.set(elapsed_time)
    if input_length != 0:
        tl_time = elapsed_time / input_length * 1000  
        api_tl_time_gauge.set(tl_time)

    memory_usage = (psutil.virtual_memory().used)/(1024**3)
    api_memory_usage_gauge.set(memory_usage)

    # CPU usage rate
    cpu_usage = psutil.cpu_percent(interval=1)
    api_cpu_usage_gauge.set(cpu_usage)

    # Network I/O bytes
    network_io_counters = psutil.net_io_counters()
    bytes_in = network_io_counters.bytes_recv
    bytes_out = network_io_counters.bytes_sent
    api_network_bytes_gauge.set((bytes_in + bytes_out)/1024)

    # Network I/O bytes rate
    bytes_rate_in = network_io_counters.bytes_recv / elapsed_time if elapsed_time > 0 else 0
    bytes_rate_out = network_io_counters.bytes_sent / elapsed_time if elapsed_time > 0 else 0
    api_network_bytes_rate_gauge.set((bytes_rate_in + bytes_rate_out)/1024)

    # returned digit is displayed
    return {"digit": digit}


