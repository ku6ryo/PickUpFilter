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
        const tipLength = v0.length()
        const distance = createVector(kp3[8], kp3[4]).length()
        const palmVector0 = createVector(kp[0], kp[1])
        const palmVector1 = createVector(kp[0], kp[9])
        const palmDirection = palmVector1.cross(palmVector0) 
        if (palmDirection.z > -0.5 && distance < tipLength * 2) {
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
    requestAnimationFrame(process)
  })
}

main()