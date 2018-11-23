import tornado.ioloop
import tornado.web
import tornado.websocket
import json
import cv2
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import dlib
from webcam import start
import socket
from multiprocessing.pool import ThreadPool
from math import floor, sqrt, ceil
import pickle

# initialize dlib's face detector
# print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
count = 0

# initializing array to store previous frames
prev_shape = []

# setting multirocessing limit as 25
pool = ThreadPool(processes=25)

# finale frame and frame on which panda will be drawn initially
blank_image = np.ones((650, 650, 3), np.uint8)*255
drawing = np.ones((650, 650, 3), np.uint8)*255


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class SimpleWebSocket(tornado.websocket.WebSocketHandler):
    connections = set()

    def open(self):
        self.connections.add(self)

    async def on_message(self, text_data):

        global prev_shape
        image = np.frombuffer(text_data, dtype=np.uint8)
        image = np.reshape(image, (600, 600, 4))

        # converted image from RGB-A to RGB
        img = image[:, :, :-1]

        # changing RGB to BGR (OpenCV uses BGR formatting)
        img = img[:, :, ::-1]

        # resizing to appropriate format and inverting it
        frame = cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC)
        frame = cv2.flip(frame, 1)

        try:
            # using multiprocessing to decrease delay in stream
            async_result = pool.apply_async(start, (frame, detector, predictor, prev_shape, blank_image, drawing))

            frame, prev_shape = async_result.get()

            # converting numpy array back to base64
            retval, frame = cv2.imencode('.jpg', frame)
            message = str(base64.b64encode(frame))[2:-1]

            await self.write_message(json.dumps({
                'message': message,
            }))
        except:
            await self.write_message(json.dumps({'message': 'error'}))

    def on_close(self):
        self.connections.remove(self)


def make_app():

    return tornado.web.Application([
        (r"/static/(.*)", tornado.web.StaticFileHandler, {'path': 'static/'}),
        (r"/static/css/(.*)", tornado.web.StaticFileHandler, {'path': 'static/'}),
        (r"/", MainHandler),
        (r"/websocket", SimpleWebSocket)
    ], autoreload=True)


if __name__ == "__main__":

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.setblocking(0)
    sockets = tornado.netutil.bind_sockets(8000)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server = tornado.httpserver.HTTPServer(make_app())
    server.add_sockets(sockets)
    tornado.ioloop.IOLoop.current().start()
