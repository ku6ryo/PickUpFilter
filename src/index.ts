import { createProgram, createShader } from "./shader";
import vertexShaderSource from "./shader.vert"
import fragmentShaderSource from "./shader.frag"
import Stats from "stats.js";
import { SupportedModels, createDetector, Keypoint } from "@tensorflow-models/hand-pose-detection"
const { MediaPipeHands } = SupportedModels
import { Vector3 } from "./Vector3";

const stats = new Stats()
document.body.appendChild(stats.dom)

function createVector(v0: Keypoint, v1: Keypoint) {
  return new Vector3(v1.x - v0.x, v1.y - v0.y, (v1.z || 0) - (v0.z || 0));
}

async function main() {
  const detector = await createDetector(MediaPipeHands, {
    runtime: "mediapipe",
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1635986972/",
  })
  const mainCanvas = document.createElement("canvas")
  const mainContext = mainCanvas.getContext("2d")
  mainCanvas.style.height = "100vh"
  mainCanvas.style.width = "100vw"
  mainCanvas.style.transform = "scale(-1, 1)"
  document.querySelector(".container")!.appendChild(mainCanvas)

  const cameraVideo = document.createElement("video");
  const cameraCanvas = document.createElement("canvas")
  const cameraContext = cameraCanvas.getContext("2d")

  const shaderCanvas = document.createElement("canvas")
  const gl = shaderCanvas.getContext("webgl")
  if (!gl) {
    throw new Error("no webgl")
  }
  if (!cameraContext) {
    throw new Error("no video context")
  }
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: {
          ideal: 960,
        },
        height: {
          ideal: 540,
        }
      },
    })
    .then(function (stream) {
      cameraVideo.srcObject = stream;
      cameraVideo.play();
    })
    .catch(function (e) {
      console.log(e)
      console.log("Something went wrong!");
    });
  } else {
    alert("getUserMedia not supported on your browser!");
  }

  const applyEffect = createApplyEffect(gl)

  async function process() {
    if (!cameraContext) {
      throw new Error("no video context")
    }
    if (!mainContext) {
      throw new Error("no main context")
    }
    stats.begin()

    cameraContext.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height)
    const hands = await detector.estimateHands(cameraCanvas)
    let effectApplied = false
    if (hands.length > 0) {
      const hand = hands[0]
      const { keypoints3D: kp3, keypoints: kp } = hand
      if (kp3) {
        const v0 = createVector(kp3[5], kp3[6])
        const v2 = createVector(kp3[7], kp3[8])
        const nv0 = v0.normalize()
        const nv2 = v2.normalize()
        const cos = nv0.dot(nv2.invert())
        const tipLength = v0.length()
        const distance = createVector(kp3[8], kp3[4]).length()
        const openness = (-cos + 1) / 2
        const palmVector0 = createVector(kp[0], kp[1])
        const palmVector1 = createVector(kp[0], kp[9])
        const palmDirection = palmVector1.cross(palmVector0) 
        if (palmDirection.z > -0.5 && distance < tipLength * 2) {
          const palm = [kp[0], kp[1], kp[9]].reduce((r, p) => {
            return {
              x: p.x / 3 + r.x,
              y: p.y / 3 + r.y,
            }
          }, { x: 0, y: 0 })
          applyEffect(cameraCanvas, palm, openness)
          mainContext.drawImage(cameraCanvas, 0, 0, mainCanvas.width, mainCanvas.height)
          mainContext.drawImage(cameraCanvas, 0, 0, mainCanvas.width, mainCanvas.height, kp[8].x, kp[8].y, mainCanvas.width / 4, mainCanvas.height / 4)
          effectApplied = true
        }
      }
    }
    if (!effectApplied) {
      mainContext.drawImage(cameraVideo, 0, 0, mainCanvas.width, mainCanvas.height)
    }
    stats.end()
    requestAnimationFrame(process)
  }
  cameraVideo.addEventListener("playing", () => {
    const vw = cameraVideo.videoWidth
    const vh = cameraVideo.videoHeight
    mainCanvas.width = vw
    mainCanvas.height = vh
    mainCanvas.style.maxHeight = `calc(100vw * ${vh / vw})`
    mainCanvas.style.maxWidth = `calc(100vh * ${vw / vh})`
    cameraCanvas.width = vw
    cameraCanvas.height = vh
    shaderCanvas.width = vw;
    shaderCanvas.height = vh;
    requestAnimationFrame(process)
  })
}

function createApplyEffect(gl: WebGLRenderingContext) {
  const vertShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
  const fragShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)
  var program = createProgram(gl, vertShader, fragShader);
  // look up where the vertex data needs to go.
  var positionLocation = gl.getAttribLocation(program, "a_position");
  var texcoordLocation = gl.getAttribLocation(program, "a_texCoord");

  // Create a buffer to put three 2d clip space points in
  var positionBuffer = gl.createBuffer();
  // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

  // time
  var timeLocation = gl.getUniformLocation(program, "u_time");

  const fingerLocation = gl.getUniformLocation(program, "u_finger");
  const openessLocation = gl.getUniformLocation(program, "u_openness");
  const resolutionLocation = gl.getUniformLocation(program, "u_resolution");

  // Tell it to use our program (pair of shaders)
  gl.useProgram(program);

  return (image: HTMLCanvasElement, finger: { x: number, y: number }, openess: number) => {
    // Create a texture.
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // Set the parameters so we can render any size image.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    // Upload the image into the texture.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Set a rectangle the same size as the image.
    setRectangle(gl, 0, 0, image.width, image.height);

    // provide texture coordinates for the rectangle.
    var texcoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      0.0,  0.0,
      1.0,  0.0,
      0.0,  1.0,
      0.0,  1.0,
      1.0,  0.0,
      1.0,  1.0,
    ]), gl.STATIC_DRAW);

    // Clear the canvas
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Turn on the position attribute
    gl.enableVertexAttribArray(positionLocation);

    // Bind the position buffer.
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    // Tell the position attribute how to get data out of positionBuffer (ARRAY_BUFFER)
    var size = 2;          // 2 components per iteration
    var type = gl.FLOAT;   // the data is 32bit floats
    var normalize = false; // don't normalize the data
    var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
    var offset = 0;        // start at the beginning of the buffer
    gl.vertexAttribPointer(
        positionLocation, size, type, normalize, stride, offset);

    // Turn on the texcoord attribute
    gl.enableVertexAttribArray(texcoordLocation);

    // bind the texcoord buffer.
    gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);

    // Tell the texcoord attribute how to get data out of texcoordBuffer (ARRAY_BUFFER)
    var size = 2;          // 2 components per iteration
    var type = gl.FLOAT;   // the data is 32bit floats
    var normalize = false; // don't normalize the data
    var stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
    var offset = 0;        // start at the beginning of the buffer
    gl.vertexAttribPointer(
        texcoordLocation, size, type, normalize, stride, offset);

    // set the resolution
    gl.uniform2f(resolutionLocation, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(fingerLocation, finger.x, finger.y);
    gl.uniform1f(openessLocation, openess);
    gl.uniform1f(timeLocation, performance.now() / 1000);

    // Draw the rectangle.
    var primitiveType = gl.TRIANGLES;
    var offset = 0;
    var count = 6;
    gl.drawArrays(primitiveType, offset, count);

    gl.deleteTexture(texture);
  }
}

function setRectangle(gl: WebGLRenderingContext, x: number, y: number, width: number, height: number) {
  var x1 = x;
  var x2 = x + width;
  var y1 = y;
  var y2 = y + height;
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
     x1, y1,
     x2, y1,
     x1, y2,
     x1, y2,
     x2, y1,
     x2, y2,
  ]), gl.STATIC_DRAW);
}

main()