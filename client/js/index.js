"use strict";
//MessageService sends messages to server using websockets
function MessageService (ws) {


    this.ws = ws;
    this.isOpen = false;
    this.onMessage = null;
    this.send = this.send.bind(this);

    ws.onopen = function () {
        this.isOpen = true
    }.bind(this);

    ws.onclose = function () {
        console.log('close');
        this.isOpen = false;
    }.bind(this);

    ws.onmessage = function (message) {
        console.log("Message", message)
        if(typeof this.onMessage === 'function'){
            console.log("Sending mes", message)
            this.onMessage(message.data);
        }
    }.bind(this)

}

MessageService.prototype.send = function(message){
    if(this.isOpen){
        this.ws.send(JSON.stringify(message));
    }
};


// Canvas has control over the canvas element
// It recieves messages from WebSocket service and draws the corresponding data
function Canvas(DOMElement) {
    //Store dom element
    this.DOMElement = DOMElement;
    this.context = this.DOMElement[0].getContext('2d');

    //Attach this to functions
    this.onMouseMove = this.onMouseMove.bind(this);
    this.drawLine = this.drawLine.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseDown = this.onMouseDown.bind(this);


    //Add mouse and touch listeners
    //TODO: Check whether some devices trigger both listeners
    this.DOMElement.on('mousedown touchstart', this.onMouseDown);
    this.DOMElement.on('mouseup touchend', this.onMouseUp);
    this.DOMElement.on('mousemove touchmove', this.onMouseMove);

    this.isMouseDown = false;
    this.prevX = null;
    this.prevy = null;

    // traceId to recognize different buffers
    this.traceId = 0;

    this.releaseTimeout = null;

    //These should be overridden
    this.onDraw = null;
    this.onComplete = null;
    this.onCompleteBuffer = null;

}

Canvas.prototype.drawLine = function (x1, y1, x2, y2) {
    this.context.lineWidth=5;
    this.context.strokeStyle="#A0A3A6";
    this.context.beginPath();
    this.context.moveTo(x1, y1);
    this.context.lineTo(x2,y2);
    this.context.closePath();
    this.context.stroke();
};


Canvas.prototype.onMouseDown = function (event) {
    event.preventDefault();
    this.isMouseDown = true;

    //Clear complete timeout from on mouse up
    if(this.releaseTimeout !== null){
        clearTimeout(this.releaseTimeout);
    }

};

Canvas.prototype.onMouseMove = function (event) {
    if(this.isMouseDown){
        event.preventDefault();
        var thisX = event.offsetX;
        var thisY = event.offsetY;

        //Handle touch event
        if(event.type === 'touchmove') {
            // https://stackoverflow.com/questions/17130940/retrieve-the-same-offsetx-on-touch-like-mouse-event

            var rect = event.target.getBoundingClientRect();
            thisX = event.targetTouches[0].pageX - rect.left;
            thisY = event.targetTouches[0].pageY - rect.top;
        }

        //Dispatch a message if onDraw is specified
        typeof this.onDraw === 'function' && this.onDraw(thisX, thisY);


        //If this is not the first press
        if(this.prevX && this.prevY){
            //Draw on canvas
            this.drawLine(this.prevX, this.prevY, thisX, thisY);


        }
        //Store last position
        this.prevX = thisX;
        this.prevY = thisY;
    }
};

Canvas.prototype.onMouseUp = function (event) {
    event.preventDefault();

    this.isMouseDown = false;

    this.prevX = undefined;
    this.prevY = undefined;

    // Tells client to end current buffer line
    // this.onCompleteBuffer(); // this function is not required anymore, since we are passing traceid in each trace through the socket.

    // increment traceid
    this.traceId++;

    console.log("On mouse up!");
    this.onComplete(this.DOMElement[0].toDataURL());


    this.releaseTimeout = setTimeout(function () {
        /*
        if(this.isMouseDown === false){
            this.onComplete(this.DOMElement[0].toDataURL());

            //Clear canvas and release timeout
            //this.context.clearRect(0,0, this.context.canvas.width, this.context.canvas.height);
            this.releaseTimeout = null;
            this.traceId = 0;
        }*/
    }.bind(this), 2000)
};


function onCompleteDrawing(uuid, dataURL, callback) {

    $.ajax({
        type: "POST",
        url: "/api",
        data: {
            uuid: uuid,
            b64_str: dataURL
        },
        contentType: 'application/x-www-form-urlencoded',
        success: function (data) {
            var parsedData = JSON.parse(data);
            callback(null, parsedData)
        }
    });
}


function initializeServer(callback) {
    $.get("/api", function (data, status) {
        if(!status === "success"){
            return callback(data);
        }
        var parsedData = JSON.parse(data);
        callback(null, parsedData);
    });
}

function handleError(error) {
    console.log("Error", error);
    alert("Noe gikk galt. Laster siden på nytt");
    location.reload();

}
$(document).ready(function () {

    var uuid = '';

    //Get canvas and prepare

    //Fill page screen
    var pageContainer = $(".page-container");
    pageContainer.css('height', window.innerHeight)

    var canvas = new Canvas($("#canvas"));

    var equation = $("#latex");
    var equationRaw = $("#latexRaw");

    var updateBtn = $("#update");

    updateBtn.click(function (e) {
        location.reload()
    });
    var awaitingMessages = 0;


    var messageService = new MessageService(new WebSocket('ws://localhost:8080/ws'));

    messageService.onMessage = function (message) {
        console.log("Got message", message);
        awaitingMessages -= 1;
        if(awaitingMessages === 0){
            updateBtn.removeClass('rotating');
        }
        katex.render(message, equation[0]);
        equationRaw.text(message)

    };

    canvas.onDraw = function (x1, y1, x2, y2) {
        messageService.send(
            {x1: x1, y1: y1, x2: x2, y2: y2,
                timestamp: performance.now(),
                uuid: uuid,
                traceid: this.traceId // states length of trace list. (since it's sent on end of current trace group)
            })
    };


    //Canvas has not been touched in 1 second
    canvas.onComplete = function (dataURL) {
        updateBtn.addClass('rotating');
        awaitingMessages += 1;
        messageService.send(
            {
                status: 201, // http 201 Created
                uuid: uuid
            }
        );
        //If a correct drawing is drawn, send a post
        onCompleteDrawing(uuid, dataURL, function (error, result) {
            // console.log(uuid, " AND ", dataURL);
            if(error){
                return handleError(error)
            }

        });
    };

    //Get initial function (TODO: Insert correct dom element)
    initializeServer(function (error, result) {
        if(error){
            return handleError(error)
        }
        console.log("Equation", result.equation);
        console.log("UUID", result.uuid);

        uuid = result.uuid;

    })
});