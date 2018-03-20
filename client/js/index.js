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
    this.erase = this.erase.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseDown = this.onMouseDown.bind(this);

    this.isErasing = false;


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
    this.didDraw = false

    this.releaseTimeout = null;

    //These should be overridden
    this.onDraw = null;
    this.onComplete = null;
    this.onCompleteBuffer = null;

}

Canvas.prototype.drawLine = function (x1, y1, x2, y2, color, lineWidth) {
    this.context.lineWidth= lineWidth ? lineWidth : 5;
    this.context.strokeStyle= color ? color : "#A0A3A6";
    this.context.beginPath();
    this.context.moveTo(x1, y1);
    this.context.lineTo(x2,y2);
    this.context.closePath();
    this.context.stroke();
};

Canvas.prototype.erase = function(x1, y1, radius) {
    radius = radius ? radius : 10
    this.context.beginPath();
    this.context.arc(x1, y1, radius, 0, 2 * Math.PI, false);
    this.context.fillStyle = 'white';
    this.context.fill();
}


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

        if(!this.didDraw){
            this.didDraw = true
        }
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
        if(this.isErasing){
            typeof this.onErase === 'function' && this.onErase(thisX, thisY);

        }
        else{
            typeof this.onDraw === 'function' && this.onDraw(thisX, thisY);
        }


        //If this is not the first press
        if(this.prevX && this.prevY){
            //Draw on canvas
            if(this.isErasing){
                this.erase(thisX, thisY);

            }
            else{
                this.drawLine(this.prevX, this.prevY, thisX, thisY);
            }


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
    if(this.didDraw){
        this.traceId++;
        this.didDraw = false
    }

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


function onCompleteDrawing(buffer, callback) {

    $.ajax({
        type: "POST",
        url: "/api",
        data: JSON.stringify({
            buffer: buffer,
        }),
        contentType: 'application/json',
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

var chartOptions = {
    type: 'bar',
    responsive: false,
    data: {
        labels: [],
        datasets: [
            {
            label: 'Sannsynligheter',
            data: [],
            
            backgroundColor: [
                'rgb(255, 99, 132)',
                'rgb(54, 162, 235)',
                'rgb(255, 206, 86)',
                'rgb(75, 192, 192)',
                'rgb(153, 102, 255)',
                'rgba(255, 159, 64, 0.2)'
            ],
        }],
        
    },
    scales: {
        xAxes: [{
            barThickness : 10
        }]
    }
};

var myNewChart = new Chart($('#chart'), chartOptions)

function displayGraphs(probabilites){
    //Number of graphs to display


    chartOptions.data.labels = probabilites.labels
    chartOptions.data.datasets[0].data = probabilites.values

    myNewChart.update()
}

function distance(x1, x2, y1, y2){
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2))
}

function getOverlapping(buffer, x, y){
    for(var [i, trace] of buffer.entries()){
        for(var coord of trace){
            if(distance(coord.x1, x, coord.y1, y) < 20){
                return i;
            }
        }
    
    }
}
function removeOverlapping(buffer, x, y, radius){
    var newBuffer = []
    console.log("Buffer", buffer)
    console.log("X", x, "Y", y, "RAD", radius)
    for(var [i, trace] of buffer.entries()){
        var filteredTraces = trace.filter(function(coord) {
            if (distance(coord.x1, x, coord.y1, y) > radius){
                console.log("Removing")
            }
            return distance(coord.x1, x, coord.y1, y) > radius
        })
        console.log("Index", i)
        console.log("tracelength", trace);
        console.log("filteredTrace", filteredTraces)
        newBuffer.push(filteredTraces)
    }
    return newBuffer;
}

$(document).ready(function () {

    var uuid = '';

    //Get canvas and prepare

    //Fill page screen
    var pageContainer = $(".page-container");
    pageContainer.css('height', window.innerHeight)

    var canvasElement = $("#canvas")
    var canvas = new Canvas(canvasElement);

    var equation = $("#latex");
    var equationRaw = $("#latexRaw");

    var updateBtn = $("#update");

    var eraseButton = $("#erase")

    updateBtn.click(function (e) {
        location.reload()
    });

    eraseButton.click(function(e) {
        canvas.isErasing = !Boolean(canvas.isErasing)
    })

    var awaitingMessages = 0;

    var probabilites = [];

    var buffer = [[]]


    var messageService = new MessageService(new WebSocket('ws://localhost:8080/ws'));

    messageService.onMessage = function (message) {
        console.log("Got message", message);
        //displayGraphs(parsedMessage.probabilites[0])

    };


    canvas.onDraw = function (x1, y1, x2, y2) {
        if(!buffer[this.traceId]) buffer.push([]);

       buffer[this.traceId].push({x1: x1, y1: y1})
       /*
       messageService.send(
            {x1: x1, y1: y1, x2: x2, y2: y2,
                timestamp: performance.now(),
                uuid: uuid,
                traceid: this.traceId // states length of trace list. (since it's sent on end of current trace group)
            }
        )*/
    };

    canvas.onErase = function (x1, y1) {
        buffer = removeOverlapping(buffer, x1, y1, 10)
        console.log("Returned new buffer", buffer[0].length)
    }

    //Canvas has not been touched in 1 second
    canvas.onComplete = function (dataURL) {
        updateBtn.addClass('rotating');
        awaitingMessages += 1;
        /*messageService.send(
            {
                status: 201, // http 201 Created
                uuid: uuid
            }
        );*/
        //If a correct drawing is drawn, send a post
        onCompleteDrawing(buffer, function (error, result) {
            if(error){
                return handleError(error)
            }
            awaitingMessages -= 1;
            if(awaitingMessages === 0){
                updateBtn.removeClass('rotating');
            }
    
            katex.render(result.latex, equation[0]);
            equationRaw.text(result.latex)
            
            probabilites = result.probabilites    

        });
    };

    canvasElement.click(function(event){
        var thisX = event.offsetX;
        var thisY = event.offsetY;

        //Handle touch event
        if(event.type === 'touchmove') {
            // https://stackoverflow.com/questions/17130940/retrieve-the-same-offsetx-on-touch-like-mouse-event

            var rect = event.target.getBoundingClientRect();
            thisX = event.targetTouches[0].pageX - rect.left;
            thisY = event.targetTouches[0].pageY - rect.top;
        }

        var traceId = getOverlapping(buffer, thisX, thisY)

        if(traceId === undefined) return;

        var traceGroup = probabilites.find(group => {
            return group.tracegroup.indexOf(traceId) >= 0
        })
        console.log("TraceId", traceId)
        console.log("Tracegroup", traceGroup)

        if(traceGroup !== undefined){
            displayGraphs(traceGroup)
            buffer.forEach((trace, i) => {
                var color = undefined;
                var strokeWidth = undefined
                if(traceGroup.tracegroup.indexOf(i) >= 0){
                    color = 'red'
                    strokeWidth = 4
                }
                for(var i = 0; i<trace.length-1;i++){
                    canvas.drawLine(trace[i].x1, trace[i].y1, trace[i+1].x1, trace[i+1].y1, color, strokeWidth)
                }
            })
        }
    })

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