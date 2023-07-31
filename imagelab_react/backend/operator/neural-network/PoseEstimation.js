const OpenCvOperator = require("../OpenCvOperator");
const fs = require("fs");
const path = require("path");
const { app } = require("electron");

/**
 * This class contains the main logic
 * of image classification
 */
class SemanticSegmentation extends OpenCvOperator {
  constructor(type, id) {
    super(type, id);
  }

  /**
   *
   * @param {Mat Image} image
   * @returns
   * Computes the AffineImage transformation
   * to the Processed Mat image
   */
  async compute(image) {

    const modelFilePath = path.resolve(app.getAppPath(), "backend", "models", "image-classification", "alexnet", "model.caffemodel");
    const configFilePath = path.resolve(app.getAppPath(), "backend", "models", "image-classification", "alexnet", "config.prototxt");
    const labelsFilePath = path.resolve(app.getAppPath(), "backend", "models", "image-classification", "alexnet", "labels.txt");

    const inputSize = [368, 368];
    const mean = [0, 0, 0];
    const std = 0.00392;
    const swapRB = false;
    const threshold = 0.1;

    // the pairs of keypoint, can be "COCO", "MPI" and "BODY_25"
    const dataset = "COCO";

    //Load model
    let net = this.cv2.readNet(configFilePath, modelFilePath);

    const input = this.getBlobFromImage(inputSize, mean, std, swapRB, image);
    net.setInput(input);
    const result = net.forward();
    const output = postProcess(result, image, threshold);

    BODY_PARTS = {};
    POSE_PAIRS = [];

    if (dataset === 'COCO') {
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 };

        POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    } else if (dataset === 'MPI') {
        BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                    "Background": 15 }

        POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    } else if (dataset === 'BODY_25') {
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
                    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
                    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23,
                    "RHeel": 24, "Background": 25 }

        POSE_PAIRS = [ ["Neck", "Nose"], ["Neck", "RShoulder"],
                    ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Nose", "REye"],
                    ["REye", "REar"], ["Nose", "LEye"],
                    ["LEye", "LEar"], ["Neck", "MidHip"],
                    ["MidHip", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["RAnkle", "RBigToe"],
                    ["RBigToe", "RSmallToe"], ["RAnkle", "RHeel"],
                    ["MidHip", "LHip"], ["LHip", "LKnee"],
                    ["LKnee", "LAnkle"], ["LAnkle", "LBigToe"],
                    ["LBigToe", "LSmallToe"], ["LAnkle", "LHeel"] ]
    }

    // console.log(output);
    input.delete();
    net.delete();
    result.delete();
  }

  getBlobFromImage(inputSize,mean,std,swapRB,image) {
    const mat = this.cv2.matFromImageData(image);
    let matC3= new this.cv2.Mat(mat.matSize[0],mat.matSize[1],this.cv2.CV_8UC3);
    this.cv2.cvtColor(mat,matC3,this.cv2.COLOR_RGBA2BGR);
    let input=this.cv2.blobFromImage(matC3,std,new this.cv2.Size(inputSize[0],inputSize[1]),new this.cv2.Scalar(mean[0],mean[1],mean[2]),swapRB);
    matC3.delete();
    return input;
  }

  postProcess = function(result, image, threshold) {
    const resultData = result.data32F;
    const matSize = result.matSize;
    const size1 = matSize[1];
    const size2 = matSize[2];
    const size3 = matSize[3];
    const mapSize = size2 * size3;

    const outputWidth = image.cols;
    const outputHeight = image.rows;

    let output = new cv.Mat(outputWidth, outputHeight, cv.CV_8UC3);
    cv.cvtColor(image, output, cv.COLOR_RGBA2RGB);

    // get position of keypoints from output
    let points = [];
    for (let i = 0; i < Object.keys(BODY_PARTS).length; ++i) {
        heatMap = resultData.slice(i*mapSize, (i+1)*mapSize);

        let maxIndex = 0;
        let maxConf = heatMap[0];
        for (index in heatMap) {
            if (heatMap[index] > heatMap[maxIndex]) {
                maxIndex = index;
                maxConf = heatMap[index];
            }
        }

        if (maxConf > threshold) {
            indexX = maxIndex % size3;
            indexY = maxIndex / size3;

            x = outputWidth * indexX / size3;
            y = outputHeight * indexY / size2;

            points[i] = [Math.round(x), Math.round(y)];
        }
    }

    // draw the points and lines into the image
    for (pair of POSE_PAIRS) {
        partFrom = pair[0];
        partTo = pair[1];
        idFrom = BODY_PARTS[partFrom];
        idTo = BODY_PARTS[partTo];
        pointFrom = points[idFrom];
        pointTo = points[idTo];

        if (points[idFrom] && points[idTo]) {
            cv.line(output, new cv.Point(pointFrom[0], pointFrom[1]),
                            new cv.Point(pointTo[0], pointTo[1]), new cv.Scalar(0, 255, 0), 3);
            cv.ellipse(output, new cv.Point(pointFrom[0], pointFrom[1]), new cv.Size(3, 3), 0, 0, 360,
                               new cv.Scalar(0, 0, 255), cv.FILLED);
            cv.ellipse(output, new cv.Point(pointTo[0], pointTo[1]), new cv.Size(3, 3), 0, 0, 360,
                               new cv.Scalar(0, 0, 255), cv.FILLED);
        }
    }

    return output;
  }
}

module.exports = SemanticSegmentation;
