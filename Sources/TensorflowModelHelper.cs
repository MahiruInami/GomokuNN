using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class TensorflowModelHelper
    {
        private static Random rng = new Random();
        public static void Shuffle(ref List<TrainingSample> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);

                var value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        //private static Tensor CreateResidualBlock(int filtersCount, Tensor x)
        //{
        //    var residualLayer = keras.layers.Conv2D(filtersCount, (3, 3), activation: keras.activations.Relu, padding: "same", data_format: "channels_last", strides: (1, 1)).Apply(x);
        //    residualLayer = keras.layers.BatchNormalization().Apply(residualLayer);
        //    residualLayer = keras.layers.LeakyReLU(0).Apply(residualLayer);
        //    residualLayer = keras.layers.Conv2D(filtersCount, (3, 3), activation: keras.activations.Relu, padding: "same", data_format: "channels_last", strides: (1, 1)).Apply(residualLayer);
        //    residualLayer = keras.layers.BatchNormalization().Apply(residualLayer);

        //    var mergedLayer = keras.layers.Add().Apply(new Tensor[] { residualLayer, x });
        //    mergedLayer = keras.layers.LeakyReLU(0).Apply(mergedLayer);

        //    return mergedLayer;
        //}

        //public static void BuildModels()
        //{
        //    const int FILTERS_COUNT = 256;
        //    const int RESIDUAL_BLOCKS = 12;

        //    var inputLayer = keras.layers.Input(shape: (4, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE), name: "input_layer");

        //    var netLayer = keras.layers.Conv2D(FILTERS_COUNT, (5, 5), activation: keras.activations.Relu, padding: "same", data_format: "channels_last", strides: (1, 1)).Apply(inputLayer);
        //    netLayer = keras.layers.BatchNormalization().Apply(netLayer);
        //    netLayer = keras.layers.LeakyReLU(0).Apply(netLayer);

        //    var residualLayer = CreateResidualBlock(FILTERS_COUNT, netLayer);
        //    for (int i = 0; i < RESIDUAL_BLOCKS - 1; i++)
        //    {
        //        residualLayer = CreateResidualBlock(FILTERS_COUNT, residualLayer);
        //    }

        //    var policyOutput = keras.layers.Conv2D(2, (1, 1), activation: keras.activations.Relu, padding: "same", data_format: "channels_last", strides: (1, 1)).Apply(residualLayer);
        //    policyOutput = keras.layers.BatchNormalization().Apply(policyOutput);
        //    policyOutput = keras.layers.LeakyReLU(0).Apply(policyOutput);
        //    policyOutput = keras.layers.Flatten().Apply(policyOutput);
        //    policyOutput = keras.layers.Dense(Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE, activation: keras.activations.Softmax).Apply(policyOutput);
        //    policyOutput.Name = "policy_net";

        //    var valueOutput = keras.layers.Conv2D(2, (1, 1), activation: keras.activations.Relu, padding: "same", data_format: "channels_last").Apply(residualLayer);
        //    valueOutput = keras.layers.BatchNormalization().Apply(valueOutput);
        //    valueOutput = keras.layers.LeakyReLU(0).Apply(valueOutput);
        //    valueOutput = keras.layers.Flatten().Apply(valueOutput);
        //    valueOutput = keras.layers.Dense(256, activation: keras.activations.Relu).Apply(valueOutput);
        //    valueOutput = keras.layers.Dense(1, activation: keras.activations.Tanh).Apply(valueOutput);
        //    valueOutput.Name = "value_net";

        //    var policyModel = keras.Model(inputLayer, policyOutput, name: "gomoku_policy_net");
        //    policyModel.compile(optimizer: keras.optimizers.Adam(learning_rate: 0.001f), loss: keras.losses.CategoricalCrossentropy(), metrics: new string[] { "categorical_accuracy" });

        //    policyModel.summary();
        //    policyModel.save("gomoku_policy_net_15_0.keras");

        //    var valueModel = keras.Model(inputLayer, valueOutput, name: "gomoku_value_net");
        //    valueModel.compile(optimizer: keras.optimizers.Adam(learning_rate: 0.001f), loss: keras.losses.MeanSquaredError(), metrics: new string[] { "acaccuracy" });

        //    valueModel.summary();
        //    valueModel.save("gomoku_value_net_15_0.keras");
        //}

        //public static void Train(int networkGeneration, int resultGeneration, ref List<TrainingSample> samples, float validationSplit, int batchSize, int epochCount)
        //{
        //    int validationDataAmount = (int)Math.Round(samples.Count * validationSplit);

        //    int trainingSamplesCount = samples.Count - validationDataAmount;

        //    int dataOffset = Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE;
        //    int inputDataOffset = dataOffset * 4;
        //    int inputDataSize = dataOffset * 4 * trainingSamplesCount;
        //    int policyDataSize = dataOffset * trainingSamplesCount;
        //    int valueDataSize = trainingSamplesCount;

        //    int[] inputArray = new int[inputDataSize];
        //    float[] policyArray = new float[policyDataSize];
        //    float[] valueArray = new float[valueDataSize];

        //    Shuffle(ref samples);

        //    var validationSamples = samples.GetRange(samples.Count - validationDataAmount - 1, validationDataAmount);
        //    var trainingSamples = samples.GetRange(0, trainingSamplesCount);

        //    for (int i = 0; i < trainingSamples.Count; i++)
        //    {
        //        var inputSample = trainingSamples[i].input;
        //        for (int dataIndex = 0; dataIndex < trainingSamples[i].input.Length; dataIndex++)
        //        {
        //            inputArray[i * inputDataOffset + dataIndex] = inputSample[dataIndex];
        //        }

        //        var policySample = trainingSamples[i].networkOutput;
        //        for (int dataIndex = 0; dataIndex < trainingSamples[i].networkOutput.Length; dataIndex++)
        //        {
        //            policyArray[i * dataOffset + dataIndex] = policySample[dataIndex];
        //        }

        //        valueArray[i] = trainingSamples[i].valueOutput;
        //    }


        //    int inputValidationDataSize = dataOffset * 4 * validationSamples.Count;
        //    int policyValidationDataSize = dataOffset * validationSamples.Count;
        //    int valueValidationDataSize = validationSamples.Count;

        //    int[] inputValidationArray = new int[inputValidationDataSize];
        //    float[] policyValidationArray = new float[policyValidationDataSize];
        //    float[] valueValidationArray = new float[valueValidationDataSize];

        //    for (int i = 0; i < validationSamples.Count; i++)
        //    {
        //        var inputSample = validationSamples[i].input;
        //        for (int dataIndex = 0; dataIndex < validationSamples[i].input.Length; dataIndex++)
        //        {
        //            inputValidationArray[i * inputDataOffset + dataIndex] = inputSample[dataIndex];
        //        }

        //        var policySample = validationSamples[i].networkOutput;
        //        for (int dataIndex = 0; dataIndex < validationSamples[i].networkOutput.Length; dataIndex++)
        //        {
        //            policyValidationArray[i * dataOffset + dataIndex] = policySample[dataIndex];
        //        }

        //        valueValidationArray[i] = validationSamples[i].valueOutput;
        //    }



        //    var input = np.array(inputArray).reshape((trainingSamples.Count, 4, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE));
        //    var policyOut = np.array(policyArray).reshape((trainingSamples.Count, Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE));
        //    var valueOut = np.array(valueArray).astype(np.float32);

        //    var inputValidation = np.array(inputValidationArray).reshape((validationSamples.Count, 4, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE));
        //    var policyOutValidation = np.array(policyValidationArray).reshape((validationSamples.Count, Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE));
        //    var valueOutValidation = np.array(valueValidationArray).astype(np.float32);

        //    var policyNet = keras.models.load_model(String.Format("{0}_{1}_{2}.keras/saved_model.pb", "gomoku_policy_net", Constants.DEFAULT_BOARD_SIZE, networkGeneration));
        //    policyNet.fit(input, policyOut, batch_size: batchSize, epochs: epochCount, validation_data: (inputValidation, policyOutValidation));
        //    //var history = model.Fit(input, new NDarray[] { policyOut, valueOut }, epochs: epochCount, batch_size: batchSize, verbose: 1, validation_data_in: inputValidation, validation_data_out: new NDarray[] { policyOutValidation, valueOutValidation });

        //    policyNet.save(String.Format("{0}_{1}_{2}.keras", "gomoku_policy_net", Constants.DEFAULT_BOARD_SIZE, resultGeneration));
        //}
    }
}
