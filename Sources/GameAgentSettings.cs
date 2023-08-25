using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class GameAgentSettings
    {
        public EstimatorType type { get; set; }
        public string modelPath { get; set; }
        public int playoutsCount { get; set; }
        public float explorationRate { get; set; }
        public bool isTraining { get; set; }

        public GameAgentSettings(EstimatorType type, string modelPath, int playoutsCount, float explorationRate, bool isTraining = false)
        {
            this.type = type;
            this.modelPath = modelPath;
            this.playoutsCount = playoutsCount;
            this.explorationRate = explorationRate;
            this.isTraining = isTraining;
        }
    }
}
