const OpenCvOperator = require("../OpenCvOperator");
const fs = require("fs");
const path = require("path");
const { app } = require("electron");

/**
 * This class contains the main logic
 * of image classification
 */
class ImageClassification extends OpenCvOperator {
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
    const probs = softmax(result, needSoftmax);
    const classes = getTopClasses(probs, labels);

    console.log(classes);
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

  softmax(result, needSoftmax) {
    let arr = result.data32F;
    if (needSoftmax) {
        const maxNum = Math.max(...arr);
        const expSum = arr.map((num) => Math.exp(num - maxNum)).reduce((a, b) => a + b);
        return arr.map((value, index) => {
            return Math.exp(value - maxNum) / expSum;
        });
    } else {
        return arr;
    }
  }

  getTopClasses = function (probs,labels,topK=3){probs=Array.from(probs);let indexes=probs.map((prob,index)=>[prob,index]);let sorted=indexes.sort((a,b)=>{if(a[0]===b[0]){return 0;}
  return a[0]<b[0]?-1:1;});sorted.reverse();let classes=[];for(let i=0;i<topK;++i){let prob=sorted[i][0];let index=sorted[i][1];let c={label:labels[index],prob:(prob*100).toFixed(2)}
  classes.push(c);}
  return classes;
  }
}

module.exports = ImageClassification;
