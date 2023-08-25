using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal struct GameSettings
    {
        public GameAgentSettings firstAgent = new GameAgentSettings(EstimatorType.CNN, "", 600, 2.0f);
        public GameAgentSettings secondAgent = new GameAgentSettings(EstimatorType.NONE, "", 600, 2.0f);

        public GameSettings() { }
    }
}
