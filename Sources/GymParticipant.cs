using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class GymParticipant
    {
        public int id;
        public GameAgentSettings agent;

        public Dictionary<int, int> encountersResults = new Dictionary<int, int>();
        public Dictionary<int, int> encountersCounter = new Dictionary<int, int>();
    }
}
