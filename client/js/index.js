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
        alert(message.data);
        if(typeof this.onMessage === 'function'){
            this.onMessage(JSON.parse(message));
        }
    }.bind(this)

}

MessageService.prototype.send = function(message){
    console.log("Sending message", message)
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

        //If this is not the first press
        if(this.prevX && this.prevY){
            //Draw on canvas
            this.drawLine(this.prevX, this.prevY, thisX, thisY);

            //Then dispatch a message if onDraw is specified
            typeof this.onDraw === 'function' && this.onDraw(this.prevX, this.prevY, thisX, thisY);

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


    this.releaseTimeout = setTimeout(function () {
        if(this.isMouseDown === false){
            this.onComplete(this.DOMElement[0].toDataURL());

            //Clear canvas and release timeout
            this.context.clearRect(0,0, this.context.canvas.width, this.context.canvas.height);
            this.releaseTimeout = null;
        }
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

    var canvas = new Canvas($("#canvas"));

    var equation = $("#equation");
    console.log("EQ1", equation);
    console.log("EQ2", equation[0]);

    var messageService = new MessageService(new WebSocket('ws://localhost:8080/ws'));

    canvas.onDraw = function (x1, y1, x2, y2) {
        messageService.send(
            {x1: x1, y1: y1, x2: x2, y2: y2,
                timestamp: performance.now(),
                uuid: uuid,
                traceid: this.traceId // states length of trace list. (since it's sent on end of current trace group)
            })
    };

    // sends an end to the server, this consists of an msg and an id. id represents trace id in trace group
    // TODO Delete this ( ? ) (this method shouldnt be needed anymore, validate.)
    /*canvas.onCompleteBuffer = function () {
        messageService.send(
            {
                status: 201, // http 201 Created
                traceid: this.traceId,
                uuid: uuid
            }
        )
    };*/



    //Canvas has not been touched in 1 second
    canvas.onComplete = function (dataURL) {
        equation.text("");
        messageService.send(
            {
                status: 201, // http 201 Created
                traceid: this.traceId,
                uuid: uuid
            }
        );
        //If a correct drawing is drawn, send a post
        onCompleteDrawing(uuid, dataURL, function (error, result) {
            // console.log(uuid, " AND ", dataURL);
            if(error){
                return handleError(error)
            }

            equation.text(result.equation);

        });
    };

    //Get initial function (TODO: Insert correct dom element)
    initializeServer(function (error, result) {
        if(error){
            return handleError(error)
        }
        console.log("Equation", result.equation);
        console.log("UUID", result.uuid);

        equation.text(result.equation);
        uuid = result.uuid;

    })
});