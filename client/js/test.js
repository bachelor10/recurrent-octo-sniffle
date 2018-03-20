function distance(x1, x2, y1, y2){
    return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2))
}

function getOverlapping(buffer, x, y){
    for(var [i, trace] of buffer.entries()){
        for(var coord of trace){
            if(distance(coord.x, x, coord.y, y) < 1.5){
                return i;
            }
        }
    
    }
}


const b = [
    [{x: 1, y: 1},{x: 2, y: 1},{x: 3, y: 1},{x: 4, y: 1},{x: 5, y: 1}],
    [{x: 1, y: 3},{x: 2, y: 3},{x: 3, y: 3},{x: 4, y: 3},{x: 5, y: 3}],
    [{x: 1, y: 5},{x: 2, y: 5},{x: 3, y: 5},{x: 4, y: 5},{x: 5, y: 5}]
]
console.log(getOverlapping(b, 3, 4))

console.log(distance(10, 5, 5, 10))