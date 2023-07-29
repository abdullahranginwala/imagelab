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

    const inputSize = [224,224];
    const mean = [104, 117, 123];
    const std = 1;
    const swapRB = false;

    // record if need softmax function for post-processing
    const needSoftmax = false;

    // Load the labels
    const labels = this.loadLabels(labelsFilePath);
    console.log(labels);

    //Load model
    let net = this.cv2.readNet(configFilePath, modelFilePath);

    const input = this.getBlobFromImage(inputSize, mean, std, swapRB, image);
    net.setInput(input);
    const result = net.forward();
    const colors = generateColors(result);
    const output = argmax(result, colors);

    console.log(output);
    input.delete();
    net.delete();
    result.delete();
  }

  loadLabels(labelsFilePath) {
    try {
      const data = fs.readFileSync(labelsFilePath, "utf8");
      const labels = data.split("\n");
      return labels;
    } catch (error) {
      console.error("Error reading labels file:", error);
      return [];
    }
  }

  getBlobFromImage(inputSize,mean,std,swapRB,image) {
    const mat = this.cv2.matFromImageData(image);
    let matC3= new this.cv2.Mat(mat.matSize[0],mat.matSize[1],this.cv2.CV_8UC3);
    this.cv2.cvtColor(mat,matC3,this.cv2.COLOR_RGBA2BGR);
    let input=this.cv2.blobFromImage(matC3,std,new this.cv2.Size(inputSize[0],inputSize[1]),new this.cv2.Scalar(mean[0],mean[1],mean[2]),swapRB);
    matC3.delete();
    return input;
  }

  generateColors(result) {
    const numClasses = result.matSize[1];
    let colors = [0,0,0];
    while(colors.length < numClasses*3){
        colors.push(Math.round((Math.random()*255 + colors[colors.length-3]) / 2));
    }
    return colors;
  }

  argmax(result, colors) {
        const C = result.matSize[1];
        const H = result.matSize[2];
        const W = result.matSize[3];
        const resultData = result.data32F;
        const imgSize = H*W;

        let classId = [];
        for (i = 0; i<imgSize; ++i) {
            let id = 0;
            for (j = 0; j < C; ++j) {
                if (resultData[j*imgSize+i] > resultData[id*imgSize+i]) {
                    id = j;
                }
            }
            classId.push(colors[id*3]);
            classId.push(colors[id*3+1]);
            classId.push(colors[id*3+2]);
            classId.push(255);
        }

        output = cv.matFromArray(H,W,cv.CV_8UC4,classId);
        return output;
  }
}

module.exports = SemanticSegmentation;
