const canvas = document.getElementById("canvas");
canvas.setAttribute('width', canvas.parentNode.offsetWidth);
canvas.setAttribute('height', canvas.parentNode.offsetHeight);

const to_settings = document.getElementById("to_settings");
const offcanvasBody = document.getElementById("offcanvas-body");
var	mouseClicked = false;
var prevX = 0;
var currX = 0;
var prevY = 0;
var currY = 0;
var globalCompositeOperation = "source-over";



fillStyle = document.getElementById("input_color").value
lineWidth = parseInt(document.getElementById("input_linewidth").value)
// canvas.addEventListener("mousemove", handleMouseEvent);
// canvas.addEventListener("mousedown", handleMouseEvent);
// canvas.addEventListener("mouseup", handleMouseEvent);
// canvas.addEventListener("mouseout", handleMouseEvent);


function handleMouseEvent(e) {
    if (e.type === 'mousedown') {
      prevX = currX;
      prevY = currY;
      currX = e.offsetX;
      currY = e.offsetY;
      mouseClicked = true;
      draw(true);
    }
    if (e.type === 'mouseup' || e.type === "mouseout") {
      mouseClicked = false;
    }
    if (e.type === 'mousemove') {
      if (mouseClicked) {
        prevX = currX;
        prevY = currY;
        currX = e.offsetX;
        currY = e.offsetY;
        draw();
      }
    }
  }
function draw(dot) {
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.globalCompositeOperation = globalCompositeOperation;
    if(dot){
      ctx.fillStyle = fillStyle;
      ctx.fillRect(currX, currY, 2, 2);
    } else {
      ctx.beginPath();
      ctx.moveTo(prevX, prevY);
      ctx.arc(currX, currY, lineWidth*3, 0, Math.PI*2, false)
      ctx.fill()
    }
    ctx.closePath();
  }

function sidebar(){
to_settings.addEventListener("mouseover", ()=>{
var offcanvasElementList = Array.prototype.slice.call(document.querySelectorAll('.offcanvas'))
var offcanvasList = offcanvasElementList.map(function (offcanvasEl) {
  return new bootstrap.Offcanvas(offcanvasEl)
})

offcanvasList.forEach(element => {
  element.show()
});
});
}
function settings(){

$("#btn-upload").click(function(event) {
    img = document.getElementById("input_file").files[0]
    img_url = URL.createObjectURL(img);
    console.log(img_url)
    backgroundImage = new Image();
    backgroundImage.src = img_url;
    canvas.style.backgroundImage = "url('" + img_url + "')"
  });

$(document).on("change" , "#input_color" , function(){
fillStyle = $(this).val()
globalCompositeOperation = "source-over";
});

$(document).on("change" , "#input_linewidth" , function(){
lineWidth = $(this).val()
});

$(document).on("click", "#input_eraser", function(){

  markTool("input_eraser")
    lineWidth = parseInt(document.getElementById("input_linewidth").value)
    globalCompositeOperation = 'destination-out';
    fillStyle = "rgba(0,0,0,1)";

})

$(document).on("click", "#input_pencil", function(){
  
  markTool("input_pencil")
  canvas.addEventListener("mousemove", handleMouseEvent);
  canvas.addEventListener("mousedown", handleMouseEvent);
  canvas.addEventListener("mouseup", handleMouseEvent);
  canvas.addEventListener("mouseout", handleMouseEvent);
  fillStyle = document.getElementById("input_color").value
  lineWidth = parseInt(document.getElementById("input_linewidth").value)

})



$(document).on("click", "#input_polygon", function(){


  markTool("input_polygon")
  var canvas=document.getElementById("canvas");
  var context=canvas.getContext("2d");
  var cw=canvas.width;
  var ch=canvas.height;
  function reOffset(){
    var BB=canvas.getBoundingClientRect();
    offsetX=BB.left;
    offsetY=BB.top;        
  }
  var offsetX,offsetY;
  reOffset();
  window.onscroll=function(e){ reOffset(); }
  
  context.lineWidth=2;
  context.strokeStyle='blue';
  
  var coordinates = [];
  var isDone=false;
  
  $('#done').click(function(){
    isDone=true;
  });
  
  $("#canvas").mousedown(function(e){handleMouseDown(e);});
  
  function handleMouseDown(e){
    if(isDone || coordinates.length>20){return;}
  
    // tell the browser we're handling this event
    e.preventDefault();
    e.stopPropagation();
  
    mouseX=parseInt(e.clientX-offsetX);
    mouseY=parseInt(e.clientY-offsetY);
    coordinates.push({x:mouseX,y:mouseY});
    drawPolygon();
  }
  
  function drawPolygon(){
    // context.clearRect(0,0,cw,ch);
    context.beginPath();
    context.moveTo(coordinates[0].x, coordinates[0].y);
    for(index=1; index<coordinates.length;index++) {
      context.lineTo(coordinates[index].x, coordinates[index].y);
    }
    context.closePath();
    context.stroke();
    context.fill()
  }


})

$(document).on("click", "#input_square", function(){
  markTool('input_square')
  to_draw = false;
})
}

function markTool(tool){
  tools = ["input_square", "input_polygon", "input_pencil", "input_eraser"]
  tools.forEach(t => document.getElementById(t).style.backgroundColor = "grey")
  document.getElementById(tool).style.backgroundColor = 'blue'
  return 
}


function stage() {

  
  // comment next line to save the draw only
  // ;
  var ctx = canvas.getContext("2d");
  ctx.drawImage(canvas, 0, 0);
  maskImage = document.getElementById('maskImage')
  maskImage.src = canvas.toDataURL();
  document.getElementById("mask_download").href = canvas.toDataURL();
  document.getElementById("maskImage").style.display = "flex";

  var ctx = canvas.getContext("2d");
  ctx.drawImage(backgroundImage, 0, 0)
  backImage = document.getElementById('backImage')
  backImage.src = canvas.toDataURL();
  document.getElementById("img_download").href = canvas.toDataURL();
  document.getElementById("backImage").style.display = "flex";


  // document.getElementById("saved_img_btns").style.display='flex'
  // finalImg.style.display = "flex";

}


settings()
sidebar()