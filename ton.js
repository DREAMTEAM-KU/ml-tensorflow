const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

var csv = require("fast-csv");
var xs = [];
var ys = [];
var slice_num = 8;
var outputShape = 3

function readCSV() {
  dataArr = []
  return new Promise(function (resolve, reject) {
    csv
      .fromPath('sanam.csv', {
        delimiter: ';'
      })
      .on("data", function (str) {
        dataArr.push(str);
      })
      .on("end", function () {
        for (i of dataArr) {}
        resolve(dataArr);
      });

  });
}


async function findMax(dataSet) {
  var MAX = 0;
  // console.log(dataSet)
  for (i = 0; i < dataSet.length; i++) {
    if (MAX < Number(dataSet[i])) MAX = Number(dataSet[i]);
  }
  return MAX
}

async function to1dArray(dataSet) {
  arr = []
  for (var i = 0; i < dataSet.length; i++) {
    var row = dataSet[i];
    row = row.slice(1)
    for (var j = 0; j < row.length; j++) {
      arr.push(row[j])
    }
  }
  return arr
}

async function normaliseData(data, divider) {
  normaliseArr = []
  for (var i = 0; i < data.length; i++) {
    normaliseArr.push(data[i] / divider)
  }
  return normaliseArr
}

async function readData() {
  var csv$ = readCSV();
  var data = csv$.then(Arr => {
    return Arr
  })
  return data
}

async function reshape(data) {
  var sliced_arr = []
  for (var i = 0; i < data.length - 1; i++) {
    if (i + slice_num <= data.length) {
      sliced_arr.push(tmp_arr = data.slice(i, slice_num + i)); // Slice for first n - 1 element set            
    } else if (i + slice_num == data.length) {
      sliced_arr.push(data.slice(i))
    } else {
      break;
    }
  }
  return sliced_arr
}

async function prepareData(reshapedData) {
  freedom_set = []
  follow_set = []
  for (var i = 0; i < reshapedData.length - 1; i++) {
    nthRow = reshapedData[i]
    freedom_set.push(nthRow.slice(0, slice_num - outputShape))
    follow_set.push(nthRow.slice(slice_num - outputShape))
  }
  return [freedom_set, follow_set]
}



var model = tf.sequential();

// input of prediction
model.add(tf.layers.lstm({
  units: 200,
  inputShape: [slice_num - outputShape, 1],
  returnSequences: false
}));

// output of prediction
model.add(tf.layers.dense({
  units: 3, // จำนวน output 
  kernelInitializer: 'VarianceScaling',
  activation: 'relu'
}));

const LEARNING_RATE = 0.001;
const optimizer = tf.train.adam(LEARNING_RATE);

model.compile({
  optimizer: optimizer,
  loss: 'meanSquaredError',
  metrics: ['accuracy'],
});

async function prepareTestSet(data, percent) {
  i = Math.floor(data.length * (percent / 100))
  // console.log(i)
  return data.slice(-i)
}

async function prepareTrainSet(data, percent) {
  i = Math.floor(data.length * (percent / 100))
  // console.log(i)
  return data.slice(0, i)
}

async function main() {
  async function trainModel() {
    const history = await model.fit(
      trainXS,
      trainYS, {
        batchSize: 1, // element in array 
        epochs: 1, // number of round to train 
        shuffle: false,
        validationSplit: 0.2
      });
  }
  var rawData = await readData();
  rawData = await rawData.slice(1)
  rawData = await to1dArray(rawData)
  // console.log(rawData.length)
  var max = await findMax(rawData);
  // console.log(max)
  var normalisedData = await normaliseData(rawData, max);
  // console.log(normalisedData)
  var reshapedData = await reshape(normalisedData);
  // console.log(reshapedData)
  var preparedData = await prepareData(reshapedData);
  // console.log(preparedData[0])
  // console.log(preparedData[1])


  var trainXArr = await prepareTrainSet(preparedData[0], 80);
  var trainYArr = await prepareTrainSet(preparedData[1], 80);
  // // console.log(trainYArr.length)
  // console.log(trainXArr)
  // console.log(trainYArr);

  trainXS = tf.tensor2d(trainXArr);
  trainXS = tf.reshape(trainXS, [-1, slice_num - outputShape, 1])
  trainYS = tf.tensor2d(trainYArr);
  trainYS = tf.reshape(trainYS, [-1, outputShape])

  // await trainModel();
  // const saveResult = await model.save('file://model3/');

  var load = async () => {
    model = await tf.loadModel('file://model3/model.json');
  };

  await load();
  let testArr = rawData.slice(-(slice_num - outputShape))
  var a = testArr.map(function (x) {
    return parseInt(x, 10);
  });


  console.log(a)

  let testSet = tf.tensor1d(a);
  testSet = tf.reshape(testSet, [-1, slice_num - outputShape, 1]);

  let r = await model.predict(testSet);
  var result = r.dataSync();
  console.log(result);
}

// prepareData()
main();
