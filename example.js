// การบ้านคือ normalise ข้อมูลให้อยู่ 0 ถึง 1
// dropout ช่วยใ
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

var csv = require("fast-csv");
var xs = [];
var ys = [];
let data = []


function range(start, end) {
  var ans = [];
  for (let i = start; i < end; i++) {
    ans.push(i);
  }
  return ans;
}

function readCSV(file_name) {
  return new Promise(function (resolve, reject) {
    csv
      .fromPath(file_name)
      .on("data", function (str) {
        // console.log(str[0])
        data.push(str)
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

async function prepareData() {
  await readCSV('THB.csv')
  // console.log('data: ' + data)
  const len = data.length
  // console.log('data.length: ' + data.length);

  for (i = 0; i < len; i++) {
    if (MAX <= data[i][1]) {
      MAX = data[i][1];
    }
  }

  // console.log('MAX: ' + MAX)

  let dataset = data.map((number) => {
    return number[1] / MAX;
  })


  // let arr = range(Time_Step, dataset.length - NumOut + 1);
  let arr = range(0, dataset.length - 1 + 1);
  let chunk = []
  for (let i = 0; i < arr.length; i++) {
    if (i != 0 && i % 5 == 0) {
      xs.push(chunk)
      chunk = []
      ys.push([dataset[i], dataset[i + 1], dataset[i + 2]])
    }
    chunk.push(dataset[i])
  }

  console.log(xs);
  console.log(ys);

}

const model = tf.sequential();

model.add(tf.layers.lstm({
  units: 100, // จำนวน unit
  inputShape: [5, 3], // จำนวน input และ output ที่ต้องการ
  returnSequences: false // ไม่ return ผลลัพธ์ในทุกๆโหนด
}));

model.add(tf.layers.dense({
  units: 3, // จำนวน node ของ output
  kernelInitializer: 'VarianceScaling',
  activation: 'relu'
}));

const LEARNING_RATE = 0.0001;
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
        batchSize: 1, // จำนวน element ใน array ของ output
        epochs: 1, // จำนวนรอบในกสรเทรน
        shuffle: true, // สุ่มแบบเรียงหรือไม่เรียง true สุ่ม false ไม่สุ่ม
        validationSplit: 0.2 // แบ่งอัตราส่วนชุดข้อมูล test กับ train
      });
  }
  await prepareData();
  trainXS = tf.tensor2d(xs)
  trainXS = tf.reshape(trainXS, [-1, 5, 3]) // ตามจำนวนชุด data ที่นำไปเทรน,    จำนวน features , มิติของ
  trainYS = tf.tensor2d(ys)
  trainYS = tf.reshape(trainYS, [-1, 3, 1]) // [จำนวน element -1 default, จำนวนผลเฉลย 1]
  await trainModel();
  const saveResult = await model.save('file://model/');

  const load = async () => {
    const model = await tf.loadModel('file://model/model.json');
  };

  await load();

  let x = [
    0.9063013699, 0.9035616438, 0.9010958904, 0.9010958904, 0.9010958904
  ]
  let xxx = tf.tensor1d(x);
  xxx = tf.reshape(xxx, [-1, 5, 1])

  const r = await model.predict(xxx);
  let result = r.dataSync()[0];
  console.log('Scaling result: ' + result)
  console.log('result: ' + result * MAX);
}

main();
