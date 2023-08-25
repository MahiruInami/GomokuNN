using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal enum EstimatorType
    {
        NONE,
        CNN,
        MCTS,
        IDDFS,
        RANDOM
    }

    class EstimatorTypeHelper
    {
        public static EstimatorType ParseEstimatorType(string value)
        {
            return (EstimatorType)Enum.Parse(typeof(EstimatorType), value);
        }
    }
}
