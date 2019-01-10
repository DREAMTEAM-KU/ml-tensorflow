const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
let model
const load = async () => {
  model = await tf.loadModel('file://model/model.json');
};

load();

let x = [
  '9', '10', '2015'
];
let xxx = tf.tensor1d(x);
xxx = tf.reshape(xxx, [1, 3, 1])

const r = model.predict(xxx);
let result = r.dataSync()[0];
console.log(result);
