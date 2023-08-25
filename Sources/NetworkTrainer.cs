using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class NetworkTrainer
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

        public static void Train(int networkGeneration, int resultGeneration, ref List<TrainingSample> samples, float validationSplit, int batchSize, int epochCount)
        {
            int validationDataAmount = (int)Math.Round(samples.Count * validationSplit);

            int trainingSamplesCount = samples.Count - validationDataAmount;

            int dataOffset = Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE;
            int inputDataOffset = dataOffset * 4;
            int inputDataSize = dataOffset * 4 * trainingSamplesCount;
            int policyDataSize = dataOffset * trainingSamplesCount;
            int valueDataSize = trainingSamplesCount;

            int[] inputArray = new int[inputDataSize];
            float[] policyArray = new float[policyDataSize];
            float[] valueArray = new float[valueDataSize];

            Shuffle(ref samples);

            var validationSamples = samples.GetRange(samples.Count - validationDataAmount - 1, validationDataAmount);
            var trainingSamples = samples.GetRange(0, trainingSamplesCount);

            for (int i = 0; i < trainingSamples.Count; i++)
            {
                var inputSample = trainingSamples[i].input;
                for (int dataIndex = 0; dataIndex < trainingSamples[i].input.Length; dataIndex++)
                {
                    inputArray[i * inputDataOffset + dataIndex] = inputSample[dataIndex];
                }

                var policySample = trainingSamples[i].networkOutput;
                for (int dataIndex = 0; dataIndex < trainingSamples[i].networkOutput.Length; dataIndex++)
                {
                    policyArray[i * dataOffset + dataIndex] = policySample[dataIndex];
                }

                valueArray[i] = trainingSamples[i].valueOutput;
            }


            int inputValidationDataSize = dataOffset * 4 * validationSamples.Count;
            int policyValidationDataSize = dataOffset * validationSamples.Count;
            int valueValidationDataSize = validationSamples.Count;

            int[] inputValidationArray = new int[inputValidationDataSize];
            float[] policyValidationArray = new float[policyValidationDataSize];
            float[] valueValidationArray = new float[valueValidationDataSize];

            for (int i = 0; i < validationSamples.Count; i++)
            {
                var inputSample = validationSamples[i].input;
                for (int dataIndex = 0; dataIndex < validationSamples[i].input.Length; dataIndex++)
                {
                    inputValidationArray[i * inputDataOffset + dataIndex] = inputSample[dataIndex];
                }

                var policySample = validationSamples[i].networkOutput;
                for (int dataIndex = 0; dataIndex < validationSamples[i].networkOutput.Length; dataIndex++)
                {
                    policyValidationArray[i * dataOffset + dataIndex] = policySample[dataIndex];
                }

                valueValidationArray[i] = validationSamples[i].valueOutput;
            }



            var input = np.array(inputArray).reshape((trainingSamples.Count, 4, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE));
            var policyOut = np.array(policyArray).reshape((trainingSamples.Count, Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE));
            var valueOut = np.array(valueArray).astype(np.float32);

            var inputValidation = np.array(inputValidationArray).reshape((validationSamples.Count, 4, Constants.DEFAULT_BOARD_SIZE, Constants.DEFAULT_BOARD_SIZE));
            var policyOutValidation = np.array(policyValidationArray).reshape((validationSamples.Count, Constants.DEFAULT_BOARD_SIZE * Constants.DEFAULT_BOARD_SIZE));
            var valueOutValidation = np.array(valueValidationArray).astype(np.float32);

            var model = Keras.Models.Model.LoadModel(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, networkGeneration));

            var history = model.Fit(input, new NDarray[] { policyOut, valueOut }, epochs: epochCount, batch_size: batchSize, verbose: 1, validation_data_in: inputValidation, validation_data_out: new NDarray[] { policyOutValidation, valueOutValidation });

            model.Save(String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, resultGeneration));
            model.SaveOnnx(String.Format("{0}_{1}_{2}.keras.onnx", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, resultGeneration));
        }
    }
}
