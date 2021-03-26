import * as tf from "@tensorflow/tfjs";

/**
 * Build simple generator
 */
export const buildGenerator = () => {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [100], useBias: false, units: 7 * 7 * 256 }),
      tf.layers.batchNormalization(),
      tf.layers.leakyReLU(),
      tf.layers.reshape({ targetShape: [7, 7, 256] }),

      tf.layers.conv2dTranspose({ filters: 128, kernelSize: 3, padding: "same", strides: 1, useBias: false }),
      // shape - [7, 7, 128] 7x7 "matrix" where each cell is a 128 vector
      tf.layers.batchNormalization(),
      tf.layers.leakyReLU(),

      tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, padding: "same", strides: 2, useBias: false }),
      // shape - [14, 14, 64] 14x14 "matrix" where each cell is a 64 vector
      tf.layers.batchNormalization(),
      tf.layers.leakyReLU(),

      tf.layers.conv2dTranspose({ filters: 1, kernelSize: 3, padding: "same", strides: 2, useBias: false, activation: "tanh" }),
      // shape - [28, 28, 1]  28x28 matrix
    ],
  });

  return model;
};
