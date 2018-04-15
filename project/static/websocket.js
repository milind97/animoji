button = document.getElementById('start');
img = document.getElementById('image');
video = document.getElementById('camera')[0];
canvas = document.getElementById('canvas');

navigator.getMedia = (navigator.getUserMedia || navigator.webkitGetUserMedia
  || navigator.mozGetUserMedia);

//socket initialization
    var chatSocket = new WebSocket(
      'ws://' + window.location.host +
      '/websocket');

  //server response
    chatSocket.onmessage = function (e) {

      // Jsonn recieved
      var data = JSON.parse(e.data);
      var message = data['message'];
      img.src = 'data:image/png;base64,' +message;

    };

    chatSocket.onclose = function (e) {
        console.error('Chat socket closed unexpectedly');
    };

    document.querySelector('#chat-message-input').focus();
    document.querySelector('#chat-message-input').onkeyup = function (e) {
        if (e.keyCode === 13) {  // enter, return
            document.querySelector('#chat-message-submit').click();
        }
    };

    button.onclick = function (){

      navigator.getMedia(
        // constraints
        { video: true, audio: false },

        // success callback
        function (localMediaStreamTrack) {

          video.srcObject = localMediaStreamTrack;
          video.play();
        },

        //handle error
        function (error) {

          document.write("Webcam not working");
          console.log(error);
        });

        setInterval(capture_frame, 320);
    };

    var context = canvas.getContext('2d');

    function capture_frame () {

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      var json_base64 = JSON.stringify(
        {'message': canvas.toDataURL('image/jpg')})
      chatSocket.send(json_base64);
    }