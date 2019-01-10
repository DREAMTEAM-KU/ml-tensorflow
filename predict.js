// การบ้านคือ normalise ข้อมูลให้อยู่ 0 ถึง 1
// dropout ช่วยใ
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const axios = require('axios');

const express = require('express')
const app = express()
const port = 4000

let url = 'http://tesatopgun.thitgorn.com/getSanam?hours='
// let url = 'http://localhot/getSanam?hours='

let data = []

let numFeature = 10

let MAX = -999;

async function findMAX() {
  await axios.get(url + '8000').then(res => {
    data = res.data.number_of_tourist
  }).catch(err => {

  })

  MAX = Math.max(...data)

  console.log('MAX: ' + MAX)
}

const model = tf.sequential();

model.add(tf.layers.lstm({
  units: 10, // จำนวน unit
  inputShape: [numFeature, 1], // จำนวน input และ output ที่ต้องการ
  returnSequences: false // ไม่ return ผลลัพธ์ในทุกๆโหนด
}));

model.add(tf.layers.dropout({
  rate: 0.5
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

const load = async () => {
  const model = await tf.loadModel('file://model/model.json');
};


async function predict() {
  await findMAX()
  await load();
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

  let hourOne = result[0] * MAX
  let hourTwo = result[1] * MAX
  let hourThree = result[2] * MAX
  result = {
    'number_of_tourist': [hourOne, hourTwo, hourThree]
  }

  console.log(result)
  return result
}

app.get("/predict", async (req, res) => {
  // push block
  let msg = await predict()
  res.send(msg);
});

app.listen(port, () => {
  console.log(`Server running at ${port}/`);
});

// main();
