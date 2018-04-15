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

# initialize dlib's face detector
# print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
prev_shape = []
pool = ThreadPool(processes=25)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class SimpleWebSocket(tornado.websocket.WebSocketHandler):
    connections = set()

    def open(self):
        self.connections.add(self)

    async def on_message(self, text_data):

        global prev_shape
        text_data_json = json.loads(text_data)

        message = text_data_json['message'][22:]
        imgdata = base64.b64decode(message)

        image = Image.open(BytesIO(imgdata))
        img = np.array(image)[:, :, :-1]
        img = img[:, :, ::-1]

        frame = cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC)
        frame = cv2.flip(frame, 1)

        async_result = pool.apply_async(start, (frame, detector, predictor, prev_shape))
        frame, prev_shape = async_result.get()

        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        retval, frame = cv2.imencode('.jpg', frame)
        message = str(base64.b64encode(frame))[2:-1]

        [await client.write_message(json.dumps({
            'message': message,
        })) for client in self.connections]

    def on_close(self):
        self.connections.remove(self)


def make_app():

    return tornado.web.Application([
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
