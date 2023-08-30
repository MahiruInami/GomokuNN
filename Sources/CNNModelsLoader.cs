using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class CNNModelsLoader
    {
        public readonly List<string> availableModels = new List<string>();
        private readonly Config _config;

        public CNNModelsLoader(Config config) 
        {
            _config = config;

            LoadAvailableModels();
        }

        public void LoadAvailableModels()
        {
            availableModels.Clear();

            var models = Directory.GetFiles(_config.AIModelsPath);
            foreach (var modelPath in models)
            {
                if (modelPath.EndsWith(".keras") || modelPath.EndsWith(".tf") || modelPath.EndsWith(".pb"))
                {
                    availableModels.Add(modelPath);
                }
            }

            availableModels.Sort();
        }
    }
}
