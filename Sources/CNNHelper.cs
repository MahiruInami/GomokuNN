using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class CNNHelper
    {
        public static List<string> GetCNNEstimators(Config config)
        {
            var estimators = Directory.GetFiles(config.AIModelsPath);
            return estimators.ToList();
        }

        public static int GetCNNGeneration(string cnnModelPath)
        {
            if (cnnModelPath.StartsWith(Constants.MODEL_NAME))
            {
                var modelData = cnnModelPath.Split("_");
                var modelGeneration = modelData[3].Substring(0, modelData[3].Length - 6);
                return int.Parse(modelGeneration);
            }
            return 0;
        }

        public static string GetCNNPathByGeneration(int generation)
        {
            return String.Format("{0}_{1}_{2}.keras", Constants.MODEL_NAME, Constants.DEFAULT_BOARD_SIZE, generation);
        }
    }
}
