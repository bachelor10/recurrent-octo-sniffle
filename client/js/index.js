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

    this.releaseTimeout = null;

    //These should be overridden
    this.onDraw = null;
    this.onComplete = null;


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

    this.releaseTimeout = setTimeout(function () {
        if(this.isMouseDown === false){
            this.onComplete(true);
            this.releaseTimeout = null;
        }
    }.bind(this), 2000)
};


function onCompleteDrawing(callback){
    $.ajax({
        type: "POST",
        url: "/api",
        data: {
            uuid: uuid,
            b64_str: document.getElementById('canvas').toDataURL()
        },
        contentType: 'application/x-www-form-urlencoded',
        success: callback
    });

}

var uuid = '';

function setNewEquation(DOMElement) {
    $.get("/api", function(data, status){
        if(status === 'success'){
            console.log("Data", data);
            let parsedData = JSON.parse(data);
            console.log(parsedData.equation);
            console.log(parsedData.uuid);
            uuid = parsedData.uuid;
        }
    });
}

$(document).ready(function () {
    //Get canvas and prepare

    var canvas = new Canvas($("#canvas"));

    var messageService = new MessageService(new WebSocket('ws://localhost:8080/ws'));

    canvas.onDraw = function (x1, y1, x2, y2) {
        messageService.send({x1: x1, y1: y1, x2: x2, y2: y2, timestamp: performance.now(), uuid: uuid})
    };

    canvas.onComplete = function (success) {
        //Canvas has not been touched in 1 second
        if(success){
            //If a correct drawing is drawn, send a post
            onCompleteDrawing(function (message, status) {
                //If post request was successful
                if(status === 'success'){
                    //Get and set a new equation (TODO: Input correct dom element)
                    console.log("Setting new equation");
                    setNewEquation()
                }
            });
        }
    };

    //Get initial function (TODO: Insert correct dom element)
    setNewEquation()
});