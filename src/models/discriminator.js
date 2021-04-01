import * as tf from "@tensorflow/tfjs";

/**
 * Build simple discriminator
 * @returns Discriminator model
 */
export const buildDiscriminator = () => {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({ filters: 64, kernelSize: 5, strides: 2, padding: "same", inputShape: [28, 28, 1] }),
      tf.layers.leakyReLU(),
      tf.layers.dropout({ rate: 0.3 }),

      tf.layers.conv2d({ filters: 128, kernelSize: 5, strides: 2, padding: "same" }),
      tf.layers.leakyReLU(),
      tf.layers.dropout({ rate: 0.3 }),

      tf.layers.flatten(),
      tf.layers.dense({ units: 1 }),
    ],
  });

  return model;
};
