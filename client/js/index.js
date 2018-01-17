
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


    //Add mouse listeners
    this.DOMElement.on('mousedown', this.onMouseDown);
    this.DOMElement.on('mouseup', this.onMouseUp);
    this.DOMElement.on('mousemove', this.onMouseMove);

    //Add touch listeners
    //TODO: Check whether some devices trigger both listeners
    this.DOMElement.on('touchstart', this.onMouseDown);
    this.DOMElement.on('touchend', this.onMouseUp);
    this.DOMElement.on('touchmove', this.onMouseMove);


    this.isMouseDown = false;


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

        //Draw on canvas
        this.drawLine(this.prevX, this.prevY, thisX, thisY);

        //Store last position
        this.prevX = thisX;
        this.prevY = thisY;


    }
};
Canvas.prototype.onMouseUp = function (event) {
    event.preventDefault();
    this.isMouseDown = false;
    this.prevX = undefined;
    this.prevy = undefined;
};



$(document).ready(function () {
    //Get canvas and prepare

    var canvas = new Canvas($("#canvas"));
});