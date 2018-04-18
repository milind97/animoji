``
//const  start_button = document.getElementById('start');
var img = document.getElementById('image');
var video = document.getElementById('camera');

var canvas = document.getElementById('canvas');

navigator.getMedia = (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia);

//socket initialization
var chatSocket = new WebSocket(
  'ws://' + window.location.host +
  '/websocket');

chatSocket.binaryType = 'arraybuffer';
chatSocket.bufferedAmount = 0;

var show_popup = false;
var popup = document.getElementById("myPopup");
//server response
chatSocket.onmessage = function (e) {

  // Json recieved
  //console.log("Json recieved");
  var data = JSON.parse(e.data);
  var message = data['message'];
  //console.log(message);
  if (message === 'error') {
    console.log("pop ana chayhye");
    // alert("cant detect face");
    if (!show_popup) {
      popup.classList.toggle("show");
      show_popup = true;
    }
  }
  else {
    img.src = 'data:image/png;base64,' + message;
    if (show_popup) {
      popup.classList.toggle("show");
      show_popup = false;
    }
  }
};

/*
  function Base64Encode(str, encoding = 'utf-8') {
    console.log("encodeing image");
    var bytes = new (TextEncoder || TextEncoderLite)(encoding).encode(str);
    return base64js.fromByteArray(bytes);
  }
*/
chatSocket.onclose = function (e) {
  console.error('Chat socket closed unexpectedly');
};

function start_clicked() {
  navigator.getMedia(
    // constraints
    { video: true, audio: false },

    // success callback
    function (localMediaStreamTrack) {
      var video = document.getElementsByTagName('video')[0];
      video.srcObject = localMediaStreamTrack;
      video.play();
      //console.log("video width" + video.width);
    },

    //handle error
    function (error) {

      document.write("Webcam not working");
      console.log(error);
    });

    setInterval(capture_frame, 320);
    //capture_frame();
}

//var context = canvas.getContext('2d');

var count = 0;
function capture_frame() {
  count = count + 1;
  if (count === 1 || count > 15) {
    canvas.width = video.width;
    canvas.height = video.height;
    ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var typedArray = imageData.data // data is a Uint8ClampedArray

    chatSocket.send(typedArray);
  }
}

//play pause button
function play_pause_func() {
  var play_pause = document.getElementById("play_pause");
  if (video.paused) {
    video.play();
    play_pause.innerText = "Stop";
    play_pause.style.backgroundColor = "Red";
  }
  else {
    video.pause();
    play_pause.innerText = "Play";
    play_pause.style.backgroundColor = "White";
  }
}
