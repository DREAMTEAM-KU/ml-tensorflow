// การบ้านคือ normalise ข้อมูลให้อยู่ 0 ถึง 1
// dropout ช่วยใ
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const axios = require('axios');

var csv = require("fast-csv");

let url = 'http://tesatopgun.thitgorn.com/getSanam?hours='
// let url = 'http://localhot/getSanam?hours='

var xs = [];
var ys = [];
let data = []

let numFeature = 10


function range(start, end) {
  var ans = [];
  for (let i = start; i < end; i++) {
    ans.push(i);
  }
  return ans;
}

let date = []
let fullEntry = []

function readCSV(file_name) {
  return new Promise(function (resolve, reject) {
    csv
      .fromPath(file_name)
      .on("data", function (str) {
        let splitString = str[0].split(';')
        if (splitString[0] != 'day' && splitString[0] != 'values') {
          date.push(splitString[0])
          data.push(splitString.slice(1, splitString.length))
          fullEntry.push(splitString)
        }
        // console.log(data)
        // console.log(data)
        // xs.push(str[0])
        // ys.push(str[1])
      })
      .on("end", function () {
        // console.log(csv);
        // console.log(data.length);
        resolve(data);
      });
  });
}

let MAX = -999;

async function reshape(data, slice_num) {
  var sliced_arr = []
  for (var i = 0; i < data.length - 1; i++) {
    if (i + slice_num <= data.length) {
      sliced_arr.push(tmp_arr = data.slice(i, slice_num + i)); // Slice for first n - 1 element set
      let a = data[slice_num + i]
      let b = data[slice_num + i + 1]
      let c = data[slice_num + i + 2]
      if (a === undefined) {
        a = 0
      }
      if (b === undefined) {
        b = 0
      }
      if (c === undefined) {
        c = 0
      }
      ys.push([a, b, c])
    } else if (i + slice_num == data.length) {
      sliced_arr.push(data.slice(i))
      // ys.push([data[slice_num + i], data[slice_num + i + 1], data[slice_num + i + 2]])
    } else {
      break;
    }
  }
  return sliced_arr
}

async function prepareData() {
  await axios.get(url + '8000').then(res => {
    data = res.data.number_of_tourist
    console.log(res.data.number_of_tourist)
  }).catch(err => {

  })

  // await readCSV('sanam.csv')
  // var newArr = [];
  // for (var i = 0; i < data.length; i++) {
  //   newArr = newArr.concat(data[i]);
  // }
  // data = newArr

  const len = data.length
  console.log('data.length: ' + data.length);

  MAX = Math.max(...data)

  console.log('MAX: ' + MAX)

  let dataset = data.map((number) => {
    return number / MAX;
  })

  // console.log(dataset);

  // let arr = range(Time_Step, dataset.length - NumOut + 1);
  // let arr = range(0, dataset.length - 1 + 1);

  // let chunk = []
  // for (let i = 0; i < len; i++) {
  //   if (i != 0 && i % numFeature == 0) {
  //     xs.push(chunk)
  //     chunk = []
  //     ys.push([dataset[i], dataset[i + 1], dataset[i + 2]])
  //     // ys.push(dataset[i])
  //   }
  //   chunk.push(dataset[i])
  // }

  // console.log('xs');

  // reshape(dataset, numFeature)
  xs = await reshape(dataset, numFeature)
  // ys = reshape(dataset, numFeature)

  // console.log(xs);
  // console.log(ys);

}

const model = tf.sequential();

model.add(tf.layers.lstm({
  units: 10, // จำนวน unit
  inputShape: [numFeature, 1], // จำนวน input และ output ที่ต้องการ
  returnSequences: false // ไม่ return ผลลัพธ์ในทุกๆโหนด
}));

model.add(tf.layers.dropout({
  rate: 0.2
}));

model.add(tf.layers.dense({
  units: 3, // จำนวน node ของ output
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

async function main() {
  async function trainModel() {
    const history = await model.fit(
      trainXS,
      trainYS, {
        batchSize: 20, // จำนวน element ใน array ของ output
        epochs: 15, // จำนวนรอบในกสรเทรน
        shuffle: true, // สุ่มแบบเรียงหรือไม่เรียง true สุ่ม false ไม่สุ่ม
        validationSplit: 0.2 // แบ่งอัตราส่วนชุดข้อมูล test กับ train
      });
  }
  await prepareData();
  trainXS = tf.tensor2d(xs)
  trainXS = tf.reshape(trainXS, [-1, numFeature, 1]) // ตามจำนวนชุด data ที่นำไปเทรน,    จำนวน features , มิติของ
  trainYS = tf.tensor2d(ys)
  trainYS = tf.reshape(trainYS, [-1, 3]) // [จำนวน element -1 default, จำนวนผลเฉลย 1]

  model.summary()
  await trainModel();
  const saveResult = await model.save('file://model/');

  const load = async () => {
    const model = await tf.loadModel('file://model/model.json');
  };

  await load();

  // let x = [
  //   0, 0, 0, 0.04164353137, 0.03609106052, 0.03275957801, 0.04664075514, 0.1271515825, 0.05441421433, 0.3997779012
  // ]
  let x = []

  await axios.get(url + '10').then(res => {
    x = res.data.number_of_tourist
    console.log(res.data.number_of_tourist)
  }).catch(err => {

  })

  let xxx = tf.tensor1d(x);
  xxx = tf.reshape(xxx, [-1, numFeature, 1])

  const r = await model.predict(xxx);
  let result = r.dataSync();
  console.log('Scaling result: ' + result)
  console.log('result: ' + (result[0] * MAX) + ' ' + (result[1] * MAX) + ' ' + (result[2] * MAX));
}

main();
